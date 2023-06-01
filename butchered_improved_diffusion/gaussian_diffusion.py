import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

# important utility
def _eot(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

# ModelMeanType.EPSILON
# LossType.MSE
# ModelVarType.LEARNED_RANGE

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i+1) / num_diffusion_timesteps
        betas.append(min( 1-alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):

    if schedule_name == "linear":
        scale = 100 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linsapce(beta_start, beta_end, num_diffusion_timesteps,
            dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t+0.08) / 1.008 * math.pi / 2) ** 2
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class GaussianDiffusion:
    def __init__(self, *, betas, rescale_timesteps):
        self.rescale_timesteps = rescale_timesteps
        self.betas = np.array(betas, dtype=np.float64)
        assert len(self.betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])

        alphas = 1 - betas # also 1D
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # for easy calculation for q(x_t | x_{t-1}) "1-step likelihood"
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # for easy calculation for q(x_{t-1}|x_t, x_0) "1-step posterior"
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_0, t): # address of q(x_t|x_0)
        mean = _eot(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        var = _eot(1.0 - self.alphas_cumprod, t, x_0.shape)
        logvar = _eot(self.log_one_minus_alphas_cumprod, t, x_0.shape)

        return mean, var, logvar

    def q_sample(self, x_0, t, noise=None): # actually pass through q(x_t|x_0)
        noise = th.randn_like(x_0) if noise is None else noise
        assert noise.shape == x_0.shape
        spicy_x_0 = _eot(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        spicy_noise = _eot(self.sqrt_one_minus_alphas_cumprod, t, noise.shape) * noise
        return spicy_x_0 + spicy_noise

    def q_post_mean_var(self, x_0, x_t, t): # address of q(x_{t-1}|x_t,x_0)
        assert x_0.shape == x_t.shape

        scaled_x_0 = _eot(self.posterior_mean_coef1, t, x_0.shape) * x_0
        scaled_x_t = _eot(self.posterior_mean_coef2, t, x_t.shape) * x_t
        post_mean = scaled_x_0 + scaled_x_t
        post_var = _eot(self.posterior_variance, t, x_t.shape)
        post_logvar = _eot(self.posterior_log_variance_clipped, t, x_t.shape)

        return post_mean, post_var, post_logvar

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        spicy_x_t = _eot(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        spicy_eps = _eot(self.sqrt_recipm1_alphas_cumprod, t, eps.shape) * eps
        return spicy_x_t - spicy_eps

    def p_mean_var(self, model, x_t, t, model_kwargs=None): # address of p(x_{t-1}|x_t)
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        assert model_output.shape == (B,2*C, *x_t.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)
        
        model_log_variance = model_var_values
        model_variance = th.exp(model_log_variance)

        # clamp since clip_denoised=True
        pred_x_0 = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output).clamp(-1,1)
        model_mean, _, _ = self.q_post_mean_var(x_0 = pred_x_0, x_t=x_t, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x_0": pred_x_0,
        }

    # For sampling -

    def p_sample(self, model, x_t, t, model_kwargs=None):
        out = self.p_mean_var(model, x_t, t, model_kwargs=model_kwargs)
        noise = th.randn_like(x_t)

        nonzero_mask = (t != 0).float().view(-1, *([1]*len(x_t.shape)-1)) #i.e. no noise when t==0
        sample = out["mean"] + nonzero_mask * th.exp(.5*out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def p_sample_gnrtr(self, model, shape, noise=None, model_kwargs=None):
        assert isinstance(shape, (tuple, list))
        dvc = th.device('cuda')
        x_runthru_T20 = th.randn(*shape, device=dvc) if noise is None else noise

        for i in range(self.num_timesteps, -1, -1): # [3,2,1,0]
            t = th.tensor([i] * shape[0], device=dvc)
            with th.no_grad():
                out = self.p_sample(model, x_runthru_T20, t, model_kwargs=None)
                yield out
                x_runthru_T20 = out["sample"] # to do iteratively


    def p_sample_e2e(self, model, shape, noise=None, model_kwargs=None):
        final = None

        for sample in self.p_sample_gnrtr(model,shape,noise=noise,model_kwargs=model_kwargs):
            final = sample
        
        return final["sample"]
            

    # For training - 

    def _vb_terms_bpd(self, model, x_0, t, model_kwargs=None): # in bits
        true_mean, _, true_logvar_clipped = self.q_post_mean_var(x_0=x_0, x_t=x_t,t=t)
        out = self.p_mean_var(model, x_t, t, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_logvar_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = - discretized_gaussian_log_likelihood(x_0, means=out["mean"], log_scales=.5*out["log_variance"])
        assert decoder_nll.shape == x_0.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_x_0": out["pred_x_0"]}


    def training_loss(self, model, x_0, t, model_kwargs=None, noise=None):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        noise = th.randn_like(x_0) if noise is None else noise
        x_t = self.q_sample(x_0, t, noise=noise)

        terms = {}

        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        B,C = x_t.shape[:2]
        assert model_output.shape == (B,2*C, *x_t.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)

        frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
        terms["vb"] = self._vb_terms_bpd(
            model=lambda *args, r=frozen_out: r, x_0=x_0, x_t=x_t, t=t
        )["output"]

        target = noise
        assert model_output.shape == target.shape == x_0.shape

        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"] + terms["vb"]

        return terms
        

          
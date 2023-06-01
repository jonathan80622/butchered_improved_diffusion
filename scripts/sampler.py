import argparse
import os
import inspect

import numpy as np
import torch as th

from butchered_improved_diffusion import logger
from butchered_improved_diffusion import gaussian_diffusion as gd
from butchered_improved_diffusion.respace import SpacedDiffusion, space_timesteps
from butchered_improved_diffusion.unet import UNetModel

NUM_CLASSES = 1000


# deal with argparse
def model_and_diffusion_defaults(): # just a dict
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        #learn_sigma=False,
        #sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        #use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        #rescale_learned_sigmas=True,
        #use_checkpoint=False,
        #use_scale_shift_norm=True,
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

# create diffusion object (gaussian_diffusion.py > GaussianDiffusion) 
# & model object (unet.py > UNetModel, a subclass of nn.Module)
def create_gaussian_diffusion(
    *,
    steps=1000,
    noise_schedule="linear",
    rescale_timesteps=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
    )

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


# MAIN
def main():
    #parse the args
    args = create_argparser().parse_args()

    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(th.device('cuda'))
    model.eval()

    logger.log("sampling...")
    generated_images = []
    generated_labels = []
    # sample in (args.batch_size) due to memory constraints
    # resulting in a list of (BxCxHxW) array
    while len(generated_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond: # if it's conditional diffusion
                classes = th.randint(low=0, high=NUM_CLASSES,
                    size = (args.batch_size,), device=th.device('cuda'))
                model_kwargs["y"] = classes
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        generated_images.append(sample.cpu().numpy())
        if args.class_cond:
            generated_labels.append(classes)
        logger.log(f"created {len(generated_images) * args.batch_size} samples")

    arr = np.concatenate(generated_images, axis=0)
    arr = arr[: args.num_samples] #integer problem
    if args.class_cond:
        label_arr = np.concatenate(generated_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    shape_str = "x".join([str(sh) for sh in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)
    logger.log("sampling complete")


if __name__ == "__main__":
    main()
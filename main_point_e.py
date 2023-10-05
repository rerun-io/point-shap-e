import argparse
from typing import Literal

import numpy as np
import rerun as rr
import torch
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint
from point_e.util.pc_to_mesh import marching_cubes_mesh
from tqdm.auto import tqdm


def load_sampler_and_model(
    device: torch.device, num_steps: int = 64, type: Literal["text", "img"] = "text"
):
    print("creating base model...")
    if type == "text":
        base_name = "base40M-textvec"
        model_kwargs_key_filter = ("texts", "")
        guidance_scale = [3.0, 0.0]
    elif type == "img":
        base_name = "base40M"
        model_kwargs_key_filter = ("*",)
        guidance_scale = [3.0, 3.0]
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print("creating upsample model...")
    upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

    print("creating SDF model...")
    sdf_name = "sdf"
    sdf_model = model_from_config(MODEL_CONFIGS[sdf_name], device)
    sdf_model.eval()

    print("downloading base checkpoint...")
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print("downloading upsampler checkpoint...")
    upsampler_model.load_state_dict(load_checkpoint("upsample", device))

    print("Downloading SDF model...")
    sdf_model.load_state_dict(load_checkpoint(sdf_name, device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=guidance_scale,
        karras_steps=(num_steps, num_steps),
        model_kwargs_key_filter=model_kwargs_key_filter,  # Do not condition the upsampler at all
    )
    return sampler, sdf_model


def main(prompt: str, view_steps: bool = False):
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    rr.log("prompt", rr.TextDocument(f"Prompt: {prompt}"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler, sdf_model = load_sampler_and_model(device)

    # Produce a sample from the model.
    samples = None
    diffusion_step = 0
    for x in tqdm(
        sampler.sample_batch_progressive(
            batch_size=1, model_kwargs=dict(texts=[prompt])
        )
    ):
        samples = x
        if view_steps:
            rr.set_time_sequence("diffusion_step", diffusion_step)
            if diffusion_step <= 64:
                rr.log(
                    "prompt",
                    rr.TextDocument(f"Prompt: {prompt}\nStep 1: coarse point cloud"),
                )
            if diffusion_step > 64:
                rr.log(
                    "prompt",
                    rr.TextDocument(f"Prompt: {prompt}\nStep 2: fine point cloud"),
                )

            pc = sampler.output_to_point_clouds(samples)[0]

            coords = pc.coords
            colors = np.stack([pc.channels[x] for x in "RGB"], axis=1)
            rr.log("world/points", rr.Points3D(positions=coords, colors=colors))
        diffusion_step += 1

    pc = sampler.output_to_point_clouds(samples)[0]
    coords = pc.coords
    colors = np.stack([pc.channels[x] for x in "RGB"], axis=1)
    rr.log("world/points", rr.Points3D(positions=coords, colors=colors))

    mesh = marching_cubes_mesh(
        pc=pc,
        model=sdf_model,
        batch_size=4096,
        grid_size=32,  # increase to 128 for resolution used in evals
        progress=True,
    )

    mesh_colors = np.stack([mesh.vertex_channels[x] for x in "RGB"], axis=1)
    rr.log(
        "world/mesh",
        rr.Mesh3D(
            vertex_positions=mesh.verts,
            indices=mesh.faces,
            vertex_colors=mesh_colors,
        ),
    )
    rr.log(
        "prompt",
        rr.TextDocument(f"Prompt: {prompt}\nStep 3: mesh via SDF and marching cubes"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logs rich data using the Rerun SDK.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cheeseburger",
        help="text prompt for 3d generation",
    )
    parser.add_argument(
        "--view-steps",
        action="store_true",
        help="whether to visualze diffision steps or not",
    )
    parser.add_argument("--log-diffusion", action="store_true")
    rr.script_add_args(parser)
    args, unknown = parser.parse_known_args()
    [__import__("logging").warning(f"unknown arg: {arg}") for arg in unknown]

    rr.script_setup(args, "point-e demo")
    main(args.prompt, args.log_diffusion)

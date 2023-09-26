import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from rerun.components import MeshProperties
import torch
from point_e.util.pc_to_mesh import marching_cubes_mesh
from shap_e.util.image_util import load_image
from shap_e.util.notebooks import decode_latent_mesh
from tqdm.auto import tqdm

from main_point_e import load_sampler_and_model as load_models_point_e
from main_shap_e import load_sampler_and_model as load_models_shap_e


def compare_text2mesh(prompt: str):
    shap_e_entity_path = f"shap-e/{prompt}"
    point_e_entity_path = f"point-e/{prompt}"
    rr.log(f"prompts/{prompt}", rr.TextDocument(f"Prompt: {prompt}"), timeless=True)
    rr.log(shap_e_entity_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    rr.log(point_e_entity_path, rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load in point-e and shap-e models
    point_e_sampler, sdf_model = load_models_point_e(device)
    shap_e_sampler, xm = load_models_shap_e(device, type="text", prompt=prompt)

    # Produce a sample from the model.
    samples_point_e = None
    diffusion_step = 0
    rr.set_time_sequence("diffusion_step", diffusion_step)
    diffusion_step += 1
    for x_point_e, x_shap_e in tqdm(
        zip(
            point_e_sampler.sample_batch_progressive(
                batch_size=1, model_kwargs=dict(texts=[prompt])
            ),
            shap_e_sampler.sample_batch_progressive(),
        )
    ):
        rr.set_time_sequence("diffusion_step", diffusion_step)
        # point-e generates explicit representation
        samples_point_e = x_point_e
        pc = point_e_sampler.output_to_point_clouds(samples_point_e)[0]

        coords = pc.coords
        colors = np.stack([pc.channels[x] for x in "RGB"], axis=1)
        rr.log(f"{point_e_entity_path}/points", rr.Points2D(coords, colors=colors))

        samples_shap_e = x_shap_e["x"]
        shap_e_mesh = decode_latent_mesh(xm, samples_shap_e[0]).tri_mesh()
        colors = np.stack([shap_e_mesh.vertex_channels[x] for x in "RGB"], axis=1)
        # log mesh
        rr.log(
            f"{shap_e_entity_path}/mesh",
            rr.Mesh3D(
                vertex_positions=shap_e_mesh.verts,
                mesh_properties=MeshProperties(vertex_indices=shap_e_mesh.faces),
                vertex_colors=colors,
            )
        )
        diffusion_step += 1

    mesh = marching_cubes_mesh(
        pc=pc,
        model=sdf_model,
        batch_size=4096,
        grid_size=32,  # increase to 128 for resolution used in evals
        progress=True,
    )

    mesh_colors = np.stack([mesh.vertex_channels[x] for x in "RGB"], axis=1)
    rr.log(
        f"{point_e_entity_path}/mesh",
        rr.Mesh3D(
            vertex_positions=mesh.verts,
            mesh_properties=MeshProperties(vertex_indices=mesh.faces),
            vertex_colors=mesh_colors,
        )
    )


def compare_img2mesh(image_path: Path):
    assert image_path.exists()
    rr.log("point-e", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    rr.log("shap-e", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = load_image(str(image_path))
    point_e_sampler, sdf_model = load_models_point_e(device, type="img")
    shap_e_sampler, xm = load_models_shap_e(device=device, type="img", image=image)

    # Produce a sample from the model.
    samples_point_e = None
    samples_shap_e = None
    diffusion_step = 0
    rr.set_time_sequence("diffusion_step", diffusion_step)
    rr.log_image("input_image", image)
    diffusion_step += 1
    for x_point_e, x_shap_e in tqdm(
        zip(
            point_e_sampler.sample_batch_progressive(
                batch_size=1, model_kwargs=dict(images=[image])
            ),
            shap_e_sampler.sample_batch_progressive(),
        )
    ):
        rr.set_time_sequence("diffusion_step", diffusion_step)
        # POINT-E
        samples_point_e = x_point_e
        pc = point_e_sampler.output_to_point_clouds(samples_point_e)[0]
        coords = pc.coords
        colors = np.stack([pc.channels[x] for x in "RGB"], axis=1)
        rr.log("point-e/points", rr.Points3D(coords, colors=colors))

        # SHAP-E
        samples_shap_e = x_shap_e["x"]
        shap_e_mesh = decode_latent_mesh(xm, samples_shap_e[0]).tri_mesh()
        colors = np.stack([shap_e_mesh.vertex_channels[x] for x in "RGB"], axis=1)
        # log mesh
        rr.log(
            "shap-e/mesh",
            rr.Mesh3D(
                vertex_positions=shap_e_mesh.verts,
                mesh_properties=MeshProperties(vertex_indices=shap_e_mesh.faces),
                vertex_colors=colors,
            )
        )
        diffusion_step += 1

    # Extract mesh from point-e pointcloud
    mesh = marching_cubes_mesh(
        pc=pc,
        model=sdf_model,
        batch_size=4096,
        grid_size=32,  # increase to 128 for resolution used in evals
        progress=True,
    )

    mesh_colors = np.stack([mesh.vertex_channels[x] for x in "RGB"], axis=1)
    rr.log(
        "point-e/mesh",
        rr.Mesh3D(
            vertex_positions=mesh.verts,
            mesh_properties=MeshProperties(vertex_indices=mesh.faces),
            vertex_colors=mesh_colors,
        )
    )


def main(compare_mode: Literal["text2mesh", "img2mesh"], prompt: str, image_path: Path):
    # Load in point-e and shape-3 models
    if compare_mode == "text2mesh":
        compare_text2mesh(prompt)
    elif compare_mode == "img2mesh":
        compare_img2mesh(image_path)
    else:
        for prompt in [
            "a cheeseburger",
            "a donut with pink icing",
            "a penguin",
            "ube ice cream cone",
        ]:
            compare_text2mesh(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logs rich data using the Rerun SDK.")
    parser.add_argument(
        "--compare-mode",
        type=str,
        default="text2mesh",
        choices=["text2mesh", "img2mesh", "multi-prompt"],
        help="What type of comparison to run",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cheeseburger",
        help="text prompt for 3d generation",
    )
    parser.add_argument(
        "--image-path", type=Path, default="corgi.jpg", help="path to image"
    )
    rr.script_add_args(parser)
    args, unknown = parser.parse_known_args()
    [__import__("logging").warning(f"unknown arg: {arg}") for arg in unknown]

    rr.script_setup(args, "shap-e demo")
    main(args.compare_mode, args.prompt, args.image_path)

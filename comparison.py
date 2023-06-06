import argparse
import rerun as rr
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Literal
from main_point_e import load_sampler_and_model as load_models_point_e
from main_shap_e import load_sampler_and_model as load_models_shap_e
from point_e.util.pc_to_mesh import marching_cubes_mesh
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.util.image_util import load_image


def compare_text2mesh(prompt: str):
    shap_e_log_path = f"Shap-E: {prompt}"
    point_e_log_path = f"Point-E: {prompt}"
    rr.experimental.log_text_box(f"Prompt/{prompt}", f"Prompt: {prompt}", timeless=True)
    rr.log_view_coordinates(shap_e_log_path, up="+Z", timeless=True)
    rr.log_view_coordinates(point_e_log_path, up="+Z", timeless=True)
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
        rr.log_points(f"{point_e_log_path}/points", coords, colors=colors)

        samples_shap_e = x_shap_e["x"]
        shap_e_mesh = decode_latent_mesh(xm, samples_shap_e[0]).tri_mesh()
        colors = np.stack([shap_e_mesh.vertex_channels[x] for x in "RGB"], axis=1)
        # log mesh
        rr.log_mesh(
            f"{shap_e_log_path}/mesh",
            positions=shap_e_mesh.verts,
            indices=shap_e_mesh.faces,
            vertex_colors=colors,
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
    rr.log_mesh(
        f"{point_e_log_path}/mesh",
        positions=mesh.verts,
        indices=mesh.faces,
        vertex_colors=mesh_colors,
    )


def compare_img2mesh(image_path: Path):
    assert image_path.exists()
    rr.log_view_coordinates("point-e", up="+Z", timeless=True)
    rr.log_view_coordinates("shap-e", up="+Z", timeless=True)
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
        rr.log_points("point-e/points", coords, colors=colors)

        # SHAP-E
        samples_shap_e = x_shap_e["x"]
        shap_e_mesh = decode_latent_mesh(xm, samples_shap_e[0]).tri_mesh()
        colors = np.stack([shap_e_mesh.vertex_channels[x] for x in "RGB"], axis=1)
        # log mesh
        rr.log_mesh(
            "shap-e/mesh",
            positions=shap_e_mesh.verts,
            indices=shap_e_mesh.faces,
            vertex_colors=colors,
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
    rr.log_mesh(
        "point-e/mesh",
        positions=mesh.verts,
        indices=mesh.faces,
        vertex_colors=mesh_colors,
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
    parser.add_argument("--log-diffusion", action="store_true")
    rr.script_add_args(parser)
    args, unknown = parser.parse_known_args()
    [__import__("logging").warning(f"unknown arg: {arg}") for arg in unknown]

    rr.script_setup(args, "shap-e demo")
    main(args.compare_mode, args.prompt, args.image_path)

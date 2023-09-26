import argparse
from typing import Any, Dict, Literal, Optional

import numpy as np
import rerun as rr
import torch
import torch.nn as nn
from PIL import Image
from rerun.components import MeshProperties
from shap_e.diffusion.gaussian_diffusion import GaussianDiffusion, diffusion_from_config

# from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.k_diffusion import karras_sample_progressive
from shap_e.models.download import load_config, load_model
from shap_e.models.nn.camera import DifferentiableProjectiveCamera
from shap_e.util.notebooks import (
    create_pan_cameras,
    decode_latent_images,
    decode_latent_mesh,
)


def load_sampler_and_model(
    device: torch.device,
    num_steps: int = 64,
    type: Literal["text", "img"] = "text",
    prompt: Optional[str] = "",
    image: Optional[Image.Image] = "",
):
    xm = load_model("transmitter", device=device)
    if type == "text":
        model = load_model("text300M", device=device)
        batch_size = 1
        guidance_scale = 15.0
    elif type == "img":
        model = load_model("image300M", device=device)
        batch_size = 4
        guidance_scale = 3.0
    diffusion = diffusion_from_config(load_config("diffusion"))

    model_kwargs = (
        dict(texts=[prompt] * batch_size)
        if type == "text"
        else dict(images=[image] * batch_size)
    )
    shap_es_sampler = ShapESampler(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=model_kwargs,
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        karras_steps=num_steps * 2,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    return shap_es_sampler, xm


def log_differentiable_projective_camera(
    camera: DifferentiableProjectiveCamera, idx: int
) -> None:
    # Extract translation and rotation from camera parameters
    translation = camera.origin[idx].numpy()
    # rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    rotation_matrix = np.stack(
        [camera.x[idx].numpy(), camera.y[idx].numpy(), camera.z[idx].numpy()], axis=-1
    )

    rr.log(
        "world/camera",
        rr.TranslationAndMat3x3(translation=translation, matrix=rotation_matrix),
    )

    # Compute intrinsics matrix for pinhole camera using FOV values
    width, height = camera.width, camera.height
    u_cen, v_cen = width / 2, height / 2
    f_len_x = (width / 2) / np.tan(camera.x_fov / 2)
    f_len_y = (height / 2) / np.tan(camera.y_fov / 2)
    intri = np.array(
        [[f_len_x, 0, u_cen], [0, f_len_y, v_cen], [0, 0, 1]], dtype=np.float32
    )

    # Log pinhole camera
    rr.log(
        "world/camera/image",
        rr.Pinhole(image_from_camera=intri, width=width, height=height),
    )


class ShapESampler:
    def __init__(
        self,
        batch_size: int,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        model_kwargs: Dict[str, Any],
        guidance_scale: float,
        clip_denoised: bool,
        use_fp16: bool,
        karras_steps: int,
        sigma_min: float,
        sigma_max: float,
        s_churn: float,
        device: Optional[torch.device] = None,
        progress: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.model = model
        self.diffusion = diffusion
        self.model_kwargs = model_kwargs
        self.guidance_scale = guidance_scale
        self.clip_denoised = clip_denoised
        self.use_fp16 = use_fp16
        self.karras_steps = karras_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.s_churn = s_churn
        self.device = device
        self.progress = progress

        if self.device is None:
            self.device = next(model.parameters()).device

        if hasattr(model, "cached_model_kwargs"):
            self.model_kwargs = self.model.cached_model_kwargs(
                self.batch_size, self.model_kwargs
            )
        if self.guidance_scale != 1.0 and self.guidance_scale != 0.0:
            for k, v in self.model_kwargs.copy().items():
                self.model_kwargs[k] = torch.cat([v, torch.zeros_like(v)], dim=0)

        self.sample_shape = (self.batch_size, self.model.d_latent)

    def sample_batch_progressive(self):
        samples_it = karras_sample_progressive(
            diffusion=self.diffusion,
            model=self.model,
            shape=self.sample_shape,
            steps=self.karras_steps,
            clip_denoised=self.clip_denoised,
            model_kwargs=self.model_kwargs,
            device=self.device,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            s_churn=self.s_churn,
            guidance_scale=self.guidance_scale,
            progress=self.progress,
        )
        yield from samples_it


def main(prompt: str, render_mode: str, render_size: int, view_step: bool):
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    rr.log("prompt", rr.TextDocument(f"Prompt: {prompt}"), timeless=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shap_e_sampler, xm = load_sampler_and_model(device, prompt=prompt, num_steps=32)

    latents = None
    diffusion_steps = 0
    for x in shap_e_sampler.sample_batch_progressive():
        latents = x["x"]
        if view_step:
            rr.set_time_sequence("diffusion_steps", diffusion_steps)
            mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
            colors = np.stack([mesh.vertex_channels[x] for x in "RGB"], axis=1)
            # log mesh
            rr.log(
                "world/mesh",
                rr.Mesh3D(
                    vertex_positions=mesh.verts,
                    mesh_properties=MeshProperties(vertex_indices=mesh.faces),
                    vertex_colors=colors,
                ),
            )
        diffusion_steps += 1

    # outputs form latents to visualize
    cameras = create_pan_cameras(render_size, device)
    images = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)
    final_mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
    colors = np.stack([final_mesh.vertex_channels[x] for x in "RGB"], axis=1)

    rr.log(
        "world/mesh",
        rr.Mesh3D(
            vertex_positions=final_mesh.verts,
            mesh_properties=MeshProperties(vertex_indices=final_mesh.faces),
            vertex_colors=colors,
        ),
        timeless=True
    )

    # regenerate cameras for logging
    cameras = create_pan_cameras(render_size, "cpu")

    # log images and cameras
    for idx, image in enumerate(images):
        log_differentiable_projective_camera(cameras.flat_camera, idx)
        rr.set_time_sequence("frame idx", idx)
        rr.log("Shap-E/camera/image", rr.Image(image))


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
    parser.add_argument(
        "--render-mode", choices=["stf", "nerf"], default="nerf", help="model type"
    )
    parser.add_argument(
        "--render-size", type=int, default=64, help="size of image rendered"
    )
    parser.add_argument("--log-diffusion", action="store_true")
    rr.script_add_args(parser)
    args, unknown = parser.parse_known_args()
    [__import__("logging").warning(f"unknown arg: {arg}") for arg in unknown]

    rr.script_setup(args, "shap-e demo")
    main(args.prompt, args.render_mode, args.render_size, args.log_diffusion)

import os
import random
import tempfile
import time

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from mvadapter.pipelines.pipeline_texture import TexturePipeline, ModProcessConfig
from mvadapter.utils import make_image_grid
from scripts.inference_ig2mv_sdxl import prepare_pipeline, remove_bg, run_pipeline

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
NUM_VIEWS, HEIGHT, WIDTH = 6, 1024, 1024
MAX_SEED = np.iinfo(np.int32).max

# Load pipeline
pipe = prepare_pipeline(
    base_model="lykon/dreamshaper-xl-v2-turbo",
    vae_model="madebyollin/sdxl-vae-fp16-fix",
    unet_model=None,
    lora_model=None,
    adapter_path="huanngzh/mv-adapter",
    scheduler=None,
    num_views=NUM_VIEWS,
    device=device,
    dtype=dtype,
)

texture_pipe = TexturePipeline(
    upscaler_ckpt_path="./checkpoints/4x_NMKD-Siax_200k.pth",
    inpaint_ckpt_path="./checkpoints/big-lama.pt",
    device=device,
)

# Load BiRefNet
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
).to(device)
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def infer(
        prompt, image, mesh, do_rembg=True, seed=42, randomize_seed=False,
        guidance_scale=3.0, num_inference_steps=16, reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        preprocess_mesh=False, pbr=False, upscale=True, topaz=False, texture_size=1024
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    remove_bg_fn = (
        lambda x: remove_bg(x, birefnet, transform_image, device) if do_rembg else None
    )

    images, pos_images, normal_images, preprocessed_image = run_pipeline(
        pipe=pipe,
        mesh_path=mesh.name,
        num_views=NUM_VIEWS,
        text=prompt,
        image=image,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        remove_bg_fn=remove_bg_fn,
        reference_conditioning_scale=reference_conditioning_scale,
        negative_prompt=negative_prompt,
        lora_scale=1.0,
        device=device
    )

    # Save multiview image to temporary file
    tmpdir = tempfile.mkdtemp()
    mv_path = os.path.join(tmpdir, "multiview.png")
    for i, view in enumerate(images):
        images[i].save(os.path.join(tmpdir, 'multiview_view_{}.png'.format(i)), format='PNG')
    make_image_grid(images, rows=1).save(mv_path)

    normal_path, albedo_path, orm_path = None, None, None
    if pbr:
        from mvadapter.pipelines.pipeline_pbr import generate_pbr_for_batch, RGB2XPipeline, StableNormalPipeline

        # Replace the rgb
        albedo_path = os.path.join(tmpdir, "multiview.png")
        mv_path = None

        pre_pbr_multiviews = [view.resize((1024, 1024)) for view in images[:6]]

        # Do it in batches of 6
        t0 = time.time()
        albedo_multiviews, metallic_multiviews, _, roughness_multiviews = generate_pbr_for_batch(pre_pbr_multiviews)
        t1 = time.time()

        metallic_path = os.path.join(tmpdir, "metallic.png")
        make_image_grid(metallic_multiviews, rows=1).save(metallic_path)

        roughness_path = os.path.join(tmpdir, "roughness.png")
        make_image_grid(roughness_multiviews, rows=1).save(roughness_path)

        metallic_image = Image.open(metallic_path)
        metallic_array = np.asarray(metallic_image)

        roughness_image = Image.open(roughness_path)
        roughness_array = np.asarray(roughness_image)

        orm_image = RGB2XPipeline.combine_roughness_metalness(metallic_array, roughness_array)
        orm_path = os.path.join(tmpdir, "orm.png")
        orm_image.save(orm_path)

        print(f"Generating PBR maps took {t1 - t0:.2f} seconds")

    # Generate shaded mesh
    save_name = f"mesh_output_{seed}"
    out = texture_pipe(
        mesh_path=mesh.name,
        move_to_center=True,
        preprocess_mesh=preprocess_mesh,
        save_dir=tmpdir,
        save_name=save_name,
        uv_unwarp=True,
        uv_size=texture_size,
        rgb_path=mv_path,
        rgb_process_config=ModProcessConfig(view_upscale=upscale, inpaint_mode="view"),
        base_color_path=albedo_path,
        base_color_process_config=ModProcessConfig(view_upscale=upscale, inpaint_mode="view"),
        orm_path=orm_path,
        orm_process_config=ModProcessConfig(view_upscale=False, inpaint_mode="view"),
        normal_path=normal_path,
        normal_process_config=ModProcessConfig(view_upscale=False, inpaint_mode="view"),
        camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        use_topaz=topaz,
        debug_mode=True
    )

    if out.pbr_model_save_path is not None:
        glb_path = out.pbr_model_save_path
    else:
        glb_path = out.shaded_model_save_path

    return (
        make_image_grid(images),
        make_image_grid(pos_images),
        make_image_grid(normal_images),
        preprocessed_image,
        seed,
        glb_path
    )


with gr.Blocks() as demo:
    gr.Markdown("# MV-Adapter 3D Mesh-Aware Multi-View Generator")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="high quality")
            image_input = gr.Image(label="Reference Image", type="pil")
            mesh_input = gr.File(label="Upload Mesh File (.obj/.glb/.ply)", type="filepath")
            do_rembg = gr.Checkbox(label="Remove Background", value=True)
            run_button = gr.Button("Run")

            with gr.Accordion("Advanced", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                guidance_scale = gr.Slider(label="CFG Scale", minimum=0, maximum=10, step=0.1, value=3.0)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, step=1, value=16)
                reference_conditioning_scale = gr.Slider(label="Reference Conditioning", minimum=0, maximum=2, step=0.1,
                                                         value=1.0)
                negative_prompt = gr.Textbox(label="Negative Prompt",
                                             value="watermark, ugly, deformed, noisy, blurry, low contrast")
                preprocess_mesh = gr.Checkbox(label="Pre-Process Mesh", value=False)
                texture_size = gr.Slider(minimum=1024, maximum=4096, step=1024, value=4096,
                                         label='Texture Resolution')
                upscale = gr.Checkbox(label="Upscale", value=True)
                pbr = gr.Checkbox(label="Generate PBR Maps", value=False)
                topaz = gr.Checkbox(label="Use Topaz Upscaling", value=False)

        with gr.Column():
            result = gr.Image()
            pos_output = gr.Image()
            normal_output = gr.Image()
            preprocessed_image = gr.Image()
            used_seed = gr.Textbox()
            model_output = gr.Model3D(label="Textured 3D Model")

    run_button.click(
        fn=infer,
        inputs=[
            prompt, image_input, mesh_input, do_rembg, seed, randomize_seed,
            guidance_scale, num_inference_steps, reference_conditioning_scale,
            negative_prompt,
            preprocess_mesh, pbr, upscale, topaz,texture_size
        ],
        outputs=[
            result, pos_output, normal_output, preprocessed_image, used_seed, model_output,
        ],
    )

demo.launch(server_name="0.0.0.0")

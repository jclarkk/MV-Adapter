import argparse
import os
import time

import numpy
import torch
from PIL import Image

from mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
from mvadapter.utils import make_image_grid
from .inference_tg2mv_sdxl import prepare_pipeline, run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    # I/O
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save_dir", type=str, default="./output")
    parser.add_argument("--save_name", type=str, default="t2tex_sample")
    # Extra
    parser.add_argument("--preprocess_mesh", action="store_true")
    parser.add_argument('--texture_size', type=int, default=1024)
    parser.add_argument("--upscale", action="store_true")
    parser.add_argument("--pbr", action="store_true")
    parser.add_argument('--topaz', action='store_true')
    parser.add_argument('--upscaler_path', type=str, default="./checkpoints/4x_NMKD-Siax_200k.pth")
    args = parser.parse_args()

    device = args.device
    num_views = 6

    t0 = time.time()

    # Prepare pipelines
    pipe = prepare_pipeline(
        base_model="lykon/dreamshaper-xl-v2-turbo",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler=None,
        num_views=num_views,
        device=device,
        dtype=torch.float16,
    )
    texture_pipe = TexturePipeline(
        upscaler_ckpt_path=args.upscaler_path,
        inpaint_ckpt_path="./checkpoints/big-lama.pt",
        device=device,
    )

    t1 = time.time()
    print(f"Pipeline preparation took {t1 - t0:.2f} seconds")

    os.makedirs(args.save_dir, exist_ok=True)

    t2 = time.time()
    images, pos_images, normal_images = run_pipeline(
        pipe,
        mesh_path=args.mesh,
        num_views=num_views,
        text=args.text,
        height=1024,
        width=1024,
        num_inference_steps=16,
        guidance_scale=7.0,
        seed=args.seed,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        device=device,
    )
    mv_path = os.path.join(args.save_dir, f"{args.save_name}.png")
    make_image_grid(images, rows=1).save(mv_path)

    t3 = time.time()
    print(f"Multi view image generation took {t3 - t2:.2f} seconds")

    torch.cuda.empty_cache()

    normal_path, albedo_path, orm_path = None, None, None
    if args.pbr:
        from mvadapter.pipelines.pipeline_pbr import generate_pbr_for_batch, RGB2XPipeline

        # Replace the rgb
        albedo_path = os.path.join(args.save_dir, f"{args.save_name}.png")
        mv_path = None

        pre_pbr_multiviews = [view.resize((1024, 1024)) for view in images[:6]]

        # Do it in batches of 6
        t4 = time.time()
        albedo_multiviews, metallic_multiviews, _, roughness_multiviews = generate_pbr_for_batch(pre_pbr_multiviews)

        metallic_path = os.path.join(args.save_dir, f"{args.save_name}_metallic.png")
        make_image_grid(metallic_multiviews, rows=1).save(metallic_path)

        roughness_path = os.path.join(args.save_dir, f"{args.save_name}_roughness.png")
        make_image_grid(roughness_multiviews, rows=1).save(roughness_path)

        metallic_image = Image.open(metallic_path)
        metallic_array = numpy.asarray(metallic_image)

        roughness_image = Image.open(roughness_path)
        roughness_array = numpy.asarray(roughness_image)

        orm_image = RGB2XPipeline.combine_roughness_metalness(metallic_array, roughness_array)
        orm_path = os.path.join(args.save_dir, f"{args.save_name}_orm.png")
        orm_image.save(orm_path)

        t5 = time.time()
        print(f"Generating PBR maps took {t5 - t4:.2f} seconds")

    # 2. un-project and complete texture
    t6 = time.time()
    out = texture_pipe(
        mesh_path=args.mesh,
        move_to_center=True,
        save_dir=args.save_dir,
        save_name=args.save_name,
        uv_unwarp=True,
        preprocess_mesh=args.preprocess_mesh,
        uv_size=args.texture_size,
        rgb_path=mv_path,
        rgb_process_config=ModProcessConfig(view_upscale=args.upscale, inpaint_mode="view"),
        base_color_path=albedo_path,
        base_color_process_config=ModProcessConfig(view_upscale=args.upscale, inpaint_mode="view"),
        orm_path=orm_path,
        orm_process_config=ModProcessConfig(view_upscale=False, inpaint_mode="view"),
        normal_path=normal_path,
        normal_process_config=ModProcessConfig(view_upscale=False, inpaint_mode="view"),
        camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        use_topaz=args.topaz,
    )
    t7 = time.time()
    print(f"Texture projection took {t7 - t6:.2f} seconds")

    if out.pbr_model_save_path is not None:
        glb_path = out.pbr_model_save_path
    else:
        glb_path = out.shaded_model_save_path
    print(f"Output saved to {glb_path}")

    print(f"Total time taken: {t7 - t0:.2f} seconds")

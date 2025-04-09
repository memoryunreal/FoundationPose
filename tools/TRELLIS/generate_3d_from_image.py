import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                           # 'auto' is faster but will do benchmarking at the beginning.
                                           # Recommended to set to 'native' if run only once.

import argparse
import imageio
import trimesh
import shutil
import numpy as np
import io
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate 3D model from a single image using TRELLIS')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to an optional mask image')
    parser.add_argument('--re_gen_image_path', type=str, default=None, help='Path to an optional re-generated image')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--ss_steps', type=int, default=12, help='Number of sparse structure sampler steps')
    parser.add_argument('--ss_cfg', type=float, default=7.5, help='Sparse structure sampler CFG strength')
    parser.add_argument('--slat_steps', type=int, default=12, help='Number of SLAT sampler steps')
    parser.add_argument('--slat_cfg', type=float, default=3.0, help='SLAT sampler CFG strength')
    parser.add_argument('--mesh_simplify', type=float, default=0.95, help='Mesh simplification ratio')
    parser.add_argument('--texture_size', type=int, default=1024, help='Texture size for GLB')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the pipeline
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("/nvme-ssd1/lizhe/code/FoundationPose/tools/TRELLIS/weights/TRELLIS-image-large")
    pipeline.cuda()
    
    # Load the image
    if args.re_gen_image_path is not None:
        print(f"Loading re-generated image from {args.re_gen_image_path}...")
        image = Image.open(args.re_gen_image_path)
    else:
        print(f"Loading image from {args.image_path}...")
        image = Image.open(args.image_path)
    
    # Process image with mask if provided
    if args.mask_path is not None and args.re_gen_image_path is None:
        print(f"Loading mask from {args.mask_path}...")
        mask = Image.open(args.mask_path).convert('L')
        
        # Resize mask to match image size if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)
        
        # Convert image to numpy array for processing
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Create white background
        white_bg = np.ones_like(img_array) * 255
        
        # Apply mask: keep original image where mask is non-zero, use white background elsewhere
        masked_img = np.where(mask_array[:, :, np.newaxis] > 0, img_array, white_bg)
        
        # Find the bounding box of the mask with 10-pixel padding
        nonzero_y, nonzero_x = np.where(mask_array > 0)
        if len(nonzero_y) > 0 and len(nonzero_x) > 0:
            min_y, max_y = max(0, np.min(nonzero_y) - 10), min(mask_array.shape[0], np.max(nonzero_y) + 10)
            min_x, max_x = max(0, np.min(nonzero_x) - 10), min(mask_array.shape[1], np.max(nonzero_x) + 10)
            
            # Crop the masked image
            cropped_img = masked_img[min_y:max_y, min_x:max_x]
            
            # Convert back to PIL Image
            image = Image.fromarray(cropped_img.astype(np.uint8))
            print(f"Processed image with mask and cropped to size {image.size}")
            image.save(os.path.join(args.output_dir, "cropped_image.png"))
    
    # Run the pipeline with custom parameters
    print("Generating 3D model...")
    outputs = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": args.ss_steps,
            "cfg_strength": args.ss_cfg,
        },
        slat_sampler_params={
            "steps": args.slat_steps,
            "cfg_strength": args.slat_cfg,
        },
    )
    
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    base_output_path = os.path.join(args.output_dir, base_filename)
    
    print("Saving outputs...")
    
    
    # Save 3D Gaussian as PLY
    gaussian_ply_path = f"{base_output_path}_gaussian.ply"
    outputs['gaussian'][0].save_ply(gaussian_ply_path)
    print(f"Saved 3D Gaussian to {gaussian_ply_path}")
    
    # # Convert Gaussian PLY directly to OBJ
    # obj_path = f"{base_output_path}.obj"
    # print(f"Converting PLY to OBJ: {obj_path}")
    # try:
    #     # Load the PLY file with trimesh
    #     mesh_from_ply = trimesh.load(gaussian_ply_path)
    #     # Export as OBJ
    #     mesh_from_ply.export(obj_path)
    #     print(f"Successfully exported to OBJ: {obj_path}")
    # except Exception as e:
    #     print(f"Error converting PLY to OBJ: {e}")
    
    # Render videos of different modalities
    print("Rendering videos...")
    
    # Render Gaussian video
    gaussian_video_path = f"{base_output_path}_gaussian.mp4"
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(gaussian_video_path, video, fps=30)
    print(f"Saved Gaussian video to {gaussian_video_path}")
    
    # Render Radiance Field video
    rf_video_path = f"{base_output_path}_radiance_field.mp4"
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(rf_video_path, video, fps=30)
    print(f"Saved Radiance Field video to {rf_video_path}")
    
    # Render Mesh normal video
    mesh_video_path = f"{base_output_path}_mesh_normal.mp4"
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(mesh_video_path, video, fps=30)
    print(f"Saved Mesh normal video to {mesh_video_path}")
    
    # Create and export GLB file
    print("Creating GLB file...")
    # --------------- (在 create GLB 下面多插几行) ----------------

    # 路径
    gaussian_obj_path = f"{base_output_path}_gaussian.obj"
    gaussian_mtl_path = f"{base_output_path}_gaussian.mtl"
    gaussian_tex_path = f"{base_output_path}_gaussian_texture.png"

    # 调用同样的后处理方法
    gaussian_glb, gaussian_material, gaussian_texture = postprocessing_utils.to_glb_save_texture_mtl(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=args.mesh_simplify,
        texture_size=args.texture_size,
    )

    # 导出obj
    gaussian_glb.export(gaussian_obj_path, file_type='obj')
    print(f"Saved OBJ file to {gaussian_obj_path}")

    # 导出纹理
    gaussian_texture.save(gaussian_tex_path)
    print(f"Saved gaussian texture file to {gaussian_tex_path}")

    # 写MTL
    with open(gaussian_mtl_path, 'w') as f:
        f.write(f"newmtl gaussian_material0\n")
        f.write(f"Ka 1.000000 1.000000 1.000000\n")
        f.write(f"Kd 1.000000 1.000000 1.000000\n")
        f.write(f"Ks 0.000000 0.000000 0.000000\n")
        f.write(f"Tr 1.000000\n")
        f.write(f"illum 1\n")
        f.write(f"Ns 0.000000\n")
        f.write(f"map_Kd {os.path.basename(gaussian_tex_path)}\n")

    print(f"Saved gaussian material file to {gaussian_mtl_path}")
   
    print("Done!")

if __name__ == "__main__":
    main() 
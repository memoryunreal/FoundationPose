image_path="/nvme-ssd1/lizhe/code/FoundationPose/demo_data/mustard0/rgb/1581120424100262102.png"
mask_path="/nvme-ssd1/lizhe/code/FoundationPose/demo_data/mustard0/masks/1581120424100262102.png"
re_gen_image_path="/nvme-ssd1/lizhe/code/FoundationPose/tools/TRELLIS/output/mustard0/image.png"
output_dir="/nvme-ssd1/lizhe/code/FoundationPose/tools/TRELLIS/output/mustard0"
CUDA_VISIBLE_DEVICES=1 python generate_3d_from_image.py --image_path $image_path --mask_path $mask_path --re_gen $re_gen_image_path --output_dir $output_dir

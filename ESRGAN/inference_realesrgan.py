import os
import cv2
import glob
import shutil
import time
import re
import argparse
import matplotlib.pyplot as plt

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

# ----------------- Helpers -----------------
def natural_sort(l):
    return sorted(l, key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

def display(orig, restored):
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(orig)
    ax1.set_title("Original")
    ax1.axis("off")
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(restored)
    ax2.set_title("Restored")
    ax2.axis("off")
    plt.show()

def imread_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='ESRGAN/input', help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='ESRGAN/output', help='Output folder')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance faces')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision')
    parser.add_argument('--outscale', type=float, default=4, help='Upscaling factor')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of output images')
    parser.add_argument('--tile', type=int, default=0, help='Tile size for ESRGAN')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding for ESRGAN')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding for ESRGAN')
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus', help='ESRGAN model')
    parser.add_argument('--denoise_strength', type=float, default=0.5, help='Denoise strength for x4v3 model')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU id')
    args = parser.parse_args()

    # ----------------- Paths -----------------
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    resized_dir = os.path.join(project_root, "ESRGAN/resized_temp")
    esrgan_dir = os.path.join(project_root, "ESRGAN/esrgan_raw")
    weights_dir = os.path.join(project_root, "ESRGAN/weights")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(resized_dir, exist_ok=True)
    os.makedirs(esrgan_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # ----------------- Input check -----------------
    input_list = natural_sort(glob.glob(os.path.join(input_dir, '*')))
    if not input_list:
        print(f"No input images in {input_dir}")
        return

    # ----------------- Store original sizes -----------------
    original_sizes = {}
    resized_list = []
    for img_path in input_list:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read {img_path}, skipping.")
            continue
        h, w = img.shape[:2]
        original_sizes[os.path.basename(img_path)] = (w, h)
        resized = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        resized_path = os.path.join(resized_dir, os.path.basename(img_path))
        cv2.imwrite(resized_path, resized)
        resized_list.append(resized_path)

    # ----------------- ESRGAN model -----------------
    model_name = args.model_name.split('.')[0]
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    else:
        raise ValueError(f"Model {model_name} not supported in this script")

    model_path = os.path.join(weights_dir, model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = load_file_from_url(url=file_url[0], model_dir=weights_dir, progress=True, file_name=model_name + '.pth')

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id
    )

    # ----------------- GFPGAN model -----------------
    if args.face_enhance:
        gfpgan_model_path = os.path.join(weights_dir, "GFPGANv1.3.pth")
        if not os.path.isfile(gfpgan_model_path):
            gfpgan_model_path = load_file_from_url(
                url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                model_dir=weights_dir,
                progress=True,
                file_name="GFPGANv1.3.pth"
            )
        face_enhancer = GFPGANer(
            model_path=gfpgan_model_path,
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    # ----------------- Run ESRGAN & Restore -----------------
    start_time = time.time()
    esrgan_list = []

    for img_path in resized_list:
        img = cv2.imread(img_path)
        output, _ = upsampler.enhance(img, outscale=args.outscale)
        esrgan_path = os.path.join(esrgan_dir, os.path.basename(img_path))
        cv2.imwrite(esrgan_path, output)
        esrgan_list.append(esrgan_path)

    # Restore to original sizes and apply GFPGAN if enabled
    for orig_path, esr_path in zip(input_list, esrgan_list):
        orig_img = imread_rgb(orig_path)
        esrgan_img = cv2.imread(esr_path)
        w, h = original_sizes[os.path.basename(orig_path)]
        esrgan_resized = cv2.resize(esrgan_img, (w, h), interpolation=cv2.INTER_CUBIC)

        # Face enhance if enabled
        if args.face_enhance:
            _, _, esrgan_resized = face_enhancer.enhance(esrgan_resized, has_aligned=False, only_center_face=False, paste_back=True)

        save_name = os.path.basename(orig_path).split('.')[0] + f"_{args.suffix}.png"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, esrgan_resized)

        # Optional display
        display(orig_img, cv2.cvtColor(esrgan_resized, cv2.COLOR_BGR2RGB))

    # Cleanup
    for temp_dir in [resized_dir, esrgan_dir]:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    end_time = time.time()
    print(f"\n✅ Done! Outputs saved in: {output_dir}")
    print(f"Processed {len(input_list)} images in {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()

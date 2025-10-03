import os
import cv2
import glob
import subprocess
import matplotlib.pyplot as plt
import re
import time
import shutil


def natural_sort(l):
    return sorted(l, key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

# Define paths
input_dir = os.path.join(os.getcwd(), "ESRGAN", "input")
resized_dir = os.path.join(os.getcwd(), "ESRGAN", "resized_temp")  # temp resized folder
esrgan_dir = os.path.join(os.getcwd(), "ESRGAN", "esrgan_raw")    # ESRGAN raw output
output_dir = os.path.join(os.getcwd(), "ESRGAN", "output")        # final restored output
real_esrgan_script = os.path.join(os.getcwd(), "ESRGAN", "inference_realesrgan.py")

# Ensure directories exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(resized_dir, exist_ok=True)
os.makedirs(esrgan_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Check input
input_list = natural_sort(glob.glob(os.path.join(input_dir, '*')))
if not input_list:
    print("No input images found in:", input_dir)
    exit()

# Store original sizes
original_sizes = {}

# Resize all inputs to 300x300 and save to resized_dir
resized_list = []
for img_path in input_list:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: cannot read {img_path}")
        continue
    h, w = img.shape[:2]
    original_sizes[os.path.basename(img_path)] = (w, h)  # save original size
    resized_img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    resized_path = os.path.join(resized_dir, os.path.basename(img_path))
    cv2.imwrite(resized_path, resized_img)
    resized_list.append(resized_path)

# Start timer
start_time = time.time()

# Run Real-ESRGAN on resized images
cmd = f'python "{real_esrgan_script}" -n RealESRGAN_x4plus -i "{resized_dir}" -o "{esrgan_dir}" --outscale 3.5 --face_enhance --fp32'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

if result.returncode != 0:
    print("Error running Real-ESRGAN:", result.stderr)
    exit()

# Collect ESRGAN raw outputs
esrgan_list = natural_sort(glob.glob(os.path.join(esrgan_dir, '*')))

if not esrgan_list:
    print("No images were saved by ESRGAN. Please check for errors.")
    exit()

# Function to read + convert to RGB
def imread(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        exit()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display helper
def display(img1, img2):
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title('Original Input', fontsize=16)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.title('ESRGAN Restored (Original Scale)', fontsize=16)
    ax2.axis('off')
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()

# Restore ESRGAN output back to original size & save
for input_path, esrgan_path in zip(input_list, esrgan_list):
    orig_img = imread(input_path)
    esrgan_img = cv2.imread(esrgan_path)
    if esrgan_img is None:
        print(f"Error: Unable to read ESRGAN output {esrgan_path}")
        continue

    # Resize ESRGAN output back to original size
    orig_w, orig_h = original_sizes[os.path.basename(input_path)]
    esrgan_resized = cv2.resize(esrgan_img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Save restored output
    save_path = os.path.join(output_dir, os.path.basename(input_path))
    cv2.imwrite(save_path, esrgan_resized)

    # Convert to RGB for display
    esrgan_rgb = cv2.cvtColor(esrgan_resized, cv2.COLOR_BGR2RGB)
    display(orig_img, esrgan_rgb)

print(f"\n✅ Final outputs restored to original sizes are saved in: {output_dir}")
print(f"Total processed images: {len(input_list)}")
print(f"Processing time: {elapsed_time:.2f} seconds")

# ✅ Cleanup temporary folders
for temp_dir in [resized_dir, esrgan_dir]:
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Temporary folder {temp_dir} deleted.")

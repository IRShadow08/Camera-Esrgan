import os
import cv2
import glob
import subprocess
import matplotlib.pyplot as plt
import re
import time
import threading  # for live timer

#NOTE:
#Run inference_realesrgan first to create the model weights

def natural_sort(l):
    return sorted(l, key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

# Define paths
project_dir = "C:\\Users\\User\\Desktop\\ESRGAN"
input_dir = os.path.join(project_dir, "input")
output_dir = os.path.join(project_dir, "output")
real_esrgan_script = os.path.join(project_dir, "inference_realesrgan.py")

# Ensure directories exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Detect input images
valid_exts = (".png", ".jpg", ".jpeg", ".webp")
input_list = natural_sort([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])

if not input_list:
    print("No input images found in:", input_dir)
    exit()

# --- Live Timer ---
stop_timer = False
def timer_func(start_time):
    while not stop_timer:
        elapsed = int(time.time() - start_time)
        print(f"\rProcessing... {elapsed} seconds", end="", flush=True)
        time.sleep(1)

start_time = time.time()
timer_thread = threading.Thread(target=timer_func, args=(start_time,))
timer_thread.start()

# Run Real-ESRGAN
cmd = f'python "{real_esrgan_script}" -n RealESRGAN_x4plus -i "{input_dir}" -o "{output_dir}" --outscale 3.5 --face_enhance --fp32'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

# Stop timer
stop_timer = True
timer_thread.join()
print()  # newline after timer finishes

if result.returncode != 0:
    print("Error running Real-ESRGAN:", result.stderr)
    exit()

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

# Check outputs
output_list = natural_sort([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.lower().endswith(valid_exts)])
if output_list:
    print(f"Upscaled images saved in: {output_dir}")
    print(f"Total upscaled images: {len(output_list)}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
else:
    print("No images were saved. Please check for errors.")

# --- Image Display Functions ---
def imread(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image {img_path}")
        exit()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def display(img1, img2):
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title('Input Image', fontsize=16)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.title('Real-ESRGAN Output', fontsize=16)
    ax2.axis('off')
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()

# Show before vs after
for input_path, output_path in zip(input_list, output_list):
    img_input = imread(input_path)
    img_output = imread(output_path)
    display(img_input, img_output)

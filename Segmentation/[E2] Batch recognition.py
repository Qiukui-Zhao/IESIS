import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mmseg.apis import init_model, inference_model
from tqdm import tqdm

# Configuration and model initialization
config_file = 'KNet.py'
checkpoint_file = 'checkpoint/iter_97500.pth'
device = 'cuda:0'
model = init_model(config_file, checkpoint_file, device=device)

# Directories for input and output images
input_folder = '/path/to/input/folder'
output_folder = '/path/to/output/folder'
os.makedirs(output_folder, exist_ok=True)

# Get image files from the input folder
img_files = [img_name for img_name in os.listdir(input_folder) if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process images and save segmentation masks
for img_name in tqdm(img_files, desc="Processing Images"):
    img_path = os.path.join(input_folder, img_name)
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        print(f"ERROR: {img_name} could not be loaded.")
        continue

    # Perform inference to get the segmentation result
    result = inference_model(model, img_bgr)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

    # Ensure the mask is of type uint8 for image saving
    pred_mask = (pred_mask * 255).astype(np.uint8)

    # Save the predicted mask as an image
    mask_output_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_mask.png")
    plt.imsave(mask_output_path, pred_mask, cmap='gray', format='png')

print("Identification completed!!!")

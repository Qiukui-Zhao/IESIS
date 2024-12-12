import os
import numpy as np
import matplotlib.pyplot as plt
import mmcv
import cv2
from mmseg.apis import init_model, inference_model, show_result_pyplot

# Change working directory to 'mmsegmentation'
os.chdir('mmsegmentation')

# Ensure inline plotting for Jupyter notebooks
%matplotlib inline

# Configuration and checkpoint files
config_file = 'KNet.py'
checkpoint_file = 'checkpoint/iter_97500.pth'

# Specify device (use 'cuda:0' if GPU is available, else fall back to CPU)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize the model
model = init_model(config_file, checkpoint_file, device=device)

# Load the image
img_path = '/data/1/1.jpg'
img_bgr = cv2.imread(img_path)

# Check if the image was loaded correctly
if img_bgr is None:
    raise ValueError(f"Image not found or the path is incorrect: {img_path}")

# Perform inference
result = inference_model(model, img_bgr)

# Extract prediction mask
pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

# Ensure output directory exists
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot and save the prediction mask
plt.figure(figsize=(8, 8))
plt.imshow(pred_mask, cmap='gray')
output_path = os.path.join(output_dir, 'predicted_mask.png')
plt.imsave(output_path, pred_mask, cmap='gray', format='png')
plt.show()

print(f"Prediction mask saved to {output_path}")


import os
import numpy as np
import cv2
import natsort
import xlwt
from skimage import exposure

from sceneRadianceCLAHE import RecoverCLAHE
from sceneRadianceHE import RecoverHE

np.seterr(over='ignore')

if __name__ == '__main__':
    # Define folder paths
    folder = "F:\C3\InputImages"
    input_path = os.path.join(folder, "images")
    output_path = os.path.join(folder, "clahe")

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get list of files and sort them
    files = os.listdir(input_path)
    files = natsort.natsorted(files)

    for file in files:
        filepath = os.path.join(input_path, file)
        prefix = os.path.splitext(file)[0]

        if os.path.isfile(filepath):
            print('******** Processing file ********', file)
            img = cv2.imread(filepath)

            if img is None:
                print(f"Failed to load image {file}")
                continue

            # Apply CLAHE to the image
            sceneRadiance = RecoverCLAHE(img)

            # Save the processed image
            output_filepath = os.path.join(output_path, f'{prefix}_CLAHE.jpg')
            cv2.imwrite(output_filepath, sceneRadiance)
            print(f'Saved enhanced image: {output_filepath}')

import os
import numpy as np
import cv2
import natsort
import datetime

from color_equalisation import RGB_equalisation
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')

if __name__ == '__main__':
    starttime = datetime.datetime.now()  # Initialize starttime

    folder = "C:/Users/62385/Desktop/Databases/Dataset"
    path = os.path.join(folder, "InputImages")
    files = os.listdir(path)
    files = natsort.natsorted(files)

    output_folder = 'OutputImages/'
    os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

    for file in files:
        filepath = os.path.join(path, file)
        prefix = file.split('.')[0]
        if os.path.isfile(filepath):
            print('********    file   ********', file)
            img = cv2.imread(filepath)
            if img is not None:
                sceneRadiance = RGB_equalisation(img)
                sceneRadiance = stretching(sceneRadiance)
                sceneRadiance = HSVStretching(sceneRadiance)
                sceneRadiance = sceneRadianceRGB(sceneRadiance)
                output_path = os.path.join(output_folder, f'{prefix}_UCM.jpg')
                cv2.imwrite(output_path, sceneRadiance)
            else:
                print(f'Error reading image {file}')

    endtime = datetime.datetime.now()
    time = endtime - starttime
    print('Processing time:', time)

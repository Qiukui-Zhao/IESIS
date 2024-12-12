import os
import numpy as np
import cv2
import natsort

from BL import getAtomsphericLight
from EstimateDepth import DepthMap
from getRefinedTramsmission import Refinedtransmission
from TM import getTransmission
from sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/MIP"
np.seterr(over='ignore')

if __name__ == '__main__':
    folder = "C:/Users/62385/Desktop/Databases/Dataset"
    path = folder + "/InputImages"
    files = os.listdir(path)
    files = natsort.natsorted(files)

    output_dir = 'OutputImages'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in files:
        filepath = os.path.join(path, file)
        prefix = os.path.splitext(file)[0]
        if os.path.isfile(filepath):
            print('********    Processing file   ********', file)
            img = cv2.imread(filepath)

            blockSize = 9

            largestDiff = DepthMap(img, blockSize)
            transmission = getTransmission(largestDiff)
            transmission = Refinedtransmission(transmission, img)
            AtomsphericLight = getAtomsphericLight(transmission, img)
            sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)

            cv2.imwrite(os.path.join(output_dir, f'{prefix}_MIP_TM.jpg'), np.uint8(transmission * 255))
            cv2.imwrite(os.path.join(output_dir, f'{prefix}_MIP.jpg'), sceneRadiance)
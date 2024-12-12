import os
import numpy as np
import cv2
import natsort

from RefinedTramsmission import Refinedtransmission
from getAtomsphericLight import getAtomsphericLight
from getRGBDarkChannel import getDarkChannel
from getTM import getTransmission
from sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')


def enhance_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    files = natsort.natsorted(files)

    for file in files:
        filepath = os.path.join(input_folder, file)
        prefix = os.path.splitext(file)[0]
        if os.path.isfile(filepath):
            print('******** Processing file ********', file)
            img = cv2.imread(filepath)
            blockSize = 9

            RGB_Darkchannel = getDarkChannel(img, blockSize)
            AtomsphericLight = getAtomsphericLight(RGB_Darkchannel, img)
            print('AtomsphericLight', AtomsphericLight)
            transmission = getTransmission(img, AtomsphericLight, blockSize)
            print('transmission', transmission)
            print('np.mean(transmission)', np.mean(transmission))
            transmission = Refinedtransmission(transmission, img)
            sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)

            cv2.imwrite(os.path.join(output_folder, f'{prefix}_RoWS_TM.jpg'), np.uint8(transmission * 255))
            cv2.imwrite(os.path.join(output_folder, f'{prefix}_RoWS.jpg'), sceneRadiance)


if __name__ == '__main__':
    input_folder = "C:/Users/62385/Desktop/Databases/Dataset/InputImages"
    output_folder = "C:/Users/62385/Desktop/Databases/Dataset/OutputImages"
    enhance_images(input_folder, output_folder)



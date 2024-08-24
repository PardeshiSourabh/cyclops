import pytesseract
import cv2
import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from PIL import Image
from typing import Dict, NamedTuple


class OMR(namedtuple('OMR', 'config')):
    def __new__(cls, config: Dict):
        return super(OMR, cls).__new__(cls, config=config)


class PreprocessOMR:
    def __init__(self, omr: NamedTuple):
        self.omr = omr
        self.image = cv2.imread(omr.config.get("filepath"))
        self.grayscale = self.to_grayscale()
        _, self.binarized = self.binarize()
        self.denoised = self.denoise()
        self.thickened = self.thicken_font()


    def to_grayscale(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    
    def binarize(self):
        return cv2.threshold(self.grayscale, 140, 230, cv2.THRESH_BINARY)
    

    def thicken_font(self):
        thickened = cv2.dilate(
            cv2.bitwise_not(self.binarized),
            np.ones((2,2),np.uint8),
            iterations=1
        )
        return (cv2.bitwise_not(thickened))


    def denoise(self):
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(self.binarized, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        morphed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
        blurred = cv2.medianBlur(morphed, 3)
        
        return (blurred)
    

    def display(self, type=None):
        if type == "binarized":
            img = self.binarized
        elif type == "denoised":
            img = self.denoised
        elif type == "grayscaled":
            img = self.grayscale
        elif type == "thickened":
            img = self.thickened
        else:
            img = self.image

        height, width = img.shape[:2]
        dpi = 80
        figsize = width / float(dpi), height / float(dpi)
        
        fig = plt.figure(figsize=figsize)
        
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(img, cmap='gray')
        
        plt.show()

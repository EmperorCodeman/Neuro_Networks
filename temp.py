from PIL import Image, ImageOps
import numpy as np_

color_img = Image.open('live_feed/9.png')
gray_img = ImageOps.grayscale(color_img)
n_img = np_.array( gray_img)
#x = np.array( ImageOps.grayscale( Image.open('data_sets/my handwriting/nine.png') ) ).swapaxes(1,0) #   Open image -> Image to blackandwhite(grayscale) -> img to numpy 2d array
i = 0
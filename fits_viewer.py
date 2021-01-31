# -*- coding: utf-8 -*-
"""
Author: Adriel Kim
Desc: An image viewer for fits images loaded from textfiles.
Used to get images used in technical report.
"""
from PIL import Image
import numpy as np


img_result = np.loadtxt("gpu_final_result.txt").reshape((4176, 2048))
img_OG = np.loadtxt("gpu_OG_diff.txt").reshape((4176, 2048))

img_input = np.loadtxt("inputImgOG.txt").reshape((4176, 2048))


img_result = np.interp(img_result,(img_result.min(), img_result.max()), (0, 255))
img_OG = np.interp(img_OG,(img_OG.min(), img_OG.max()), (0, 255))
img_input = np.interp(img_input,(img_input.min(), img_input.max()), (0, 255))


img_fromarr = Image.fromarray(img_result).convert('L')
img_fromarr_OG = Image.fromarray(img_OG).convert('L')
img_input_arr = Image.fromarray(img_input).convert('L')

img_fromarr.save("img_final_result.png")
img_fromarr_OG.save("img_difference_actual.png")
img_input_arr.save("input.png")

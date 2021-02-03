# -*- coding: utf-8 -*-
"""
Author: Adriel Kim
Desc: An image viewer for fits images loaded from textfiles.
Used to get images used in technical report.
"""
from PIL import Image
import numpy as np


img_result = np.loadtxt("gpu_final_result.txt").reshape((4176, 2048))#GPU OUTPUT
img_OG = np.loadtxt("gpu_OG_diff.txt").reshape((4176, 2048))#OG correction matrix
img_OG_final = np.loadtxt("finalImgOG.txt").reshape((4176, 2048))#LSST OUTPUT

img_diff = np.abs(img_OG_final-img_result)

"""
This code will crop the image to only show the max error location.

img_diff_ = img_diff[0:40,471:511]
img_OG_final = img_OG_final[0:40,471:511]
img_result = img_result[0:40,471:511]

img_diff = np.interp(img_diff, (img_diff.min(), img_diff.max()), (0, 255))#difference between GPU and CPU
img_diff_arr = Image.fromarray(img_diff).convert('L')
img_diff_arr.save("max_error_diff.png")

img_result = np.interp(img_result, (img_result.min(), img_result.max()), (0, 255))#difference between GPU and CPU
img_result_arr= Image.fromarray(img_result).convert('L')
img_result_arr.save("max_error_location_GPU.png")

img_OG_final = np.interp(img_OG_final, (img_OG_final.min(), img_OG_final.max()), (0, 255))#difference between GPU and CPU
img_OG_final_arr = Image.fromarray(img_OG_final).convert('L')
img_OG_final_arr.save("max_error_location_LSST.png")
"""


"""
The following code will convert the text file images to png for viewing
"""
img_input = np.loadtxt("inputImgOG.txt").reshape((4176, 2048))


img_result = np.interp(img_result,(img_result.min(), img_result.max()), (0, 255))
img_OG = np.interp(img_OG,(img_OG.min(), img_OG.max()), (0, 255))#difference between input image and resulting image
img_input = np.interp(img_input,(img_input.min(), img_input.max()), (0, 255))#original input image
img_diff = np.interp(img_diff, (img_diff.min(), img_diff.max()), (0, 255))#difference between GPU and CPU
img_OG_final = np.interp(img_OG_final, (img_OG_final.min(), img_OG_final.max()), (0, 255))#difference between GPU and CPU


img_fromarr = Image.fromarray(img_result).convert('L')
img_fromarr_OG = Image.fromarray(img_OG).convert('L')
img_input_arr = Image.fromarray(img_input).convert('L')
img_diff_arr = Image.fromarray(img_diff).convert('L')
img_OG_final_arr = Image.fromarray(img_OG_final).convert('L')

img_fromarr.save("img_final_result.png")
img_fromarr_OG.save("img_difference_actual.png")
img_input_arr.save("input.png")
img_diff_arr.save("difference_between_GPUandCPU.png")
img_OG_final_arr.save("LSST_ouput.png")
import IPython
import cv2
import skimage.transform
import skimage.restoration
import skimage.io
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import os, sys
import numpy as np
import skimage.transform as tr
import skimage.exposure as exposure
from skimage.filters import unsharp_mask



#
# path1 = "C:\\Users\\kuanyshov.a\\Documents\\MaskRCNN\\Model_creating_process\\data\\images\\STAMP\\data\\Blueprint_1.1\\after_pre_mask\\train"
# path2 = "C:\\Users\\kuanyshov.a\\Documents\\MaskRCNN\\Model_creating_process\\data\\images\\STAMP\\data\\Blueprint_1.1\\after_pre_mask\\"

# for file in os.listdir(path1):
#     if file.endswith(".png"):
#         im_op = Image.open(path1 + "\\" + file)
#         enhancer = ImageEnhance.Sharpness(im_op)
#         factor = 2
#         im_s_2 = enhancer.enhance(factor)
#         im_s_2.save(path2 + file)

# PREPROCESSING MASK2
# Image will be
# 1) 4096px in width and height*4096/width

# 2) high contrasted(if needed)
def pre_mask2(image):
    height, width = image.shape[:2]


    image = skimage.transform.resize(image, (int(height * 4096 / width), int(4096)))    #resising

    percentiles = np.percentile(image, (1, 99.5))                             #contrast augmentation
    # array([ 1., 28.])
    scaled = exposure.rescale_intensity(image,
                                        in_range=tuple(percentiles))
    image = unsharp_mask(scaled, radius=5, amount=2)
    image = skimage.img_as_ubyte(image)
    # image = skimage.restoration.denoise_tv_chambolle(scaled, weight=0.1)       #denoising
    return image


# path1 = "C:\\Users\\kuanyshov.a\\Documents\\Project\\Blueprints_examples\\All\\new\\val"
# path2 = "C:\\Users\\kuanyshov.a\\Documents\\MaskRCNN\\Model_creating_process\\data\\images\\STAMP\\data\\Blueprint_1.1\\after_pre_mask\\val\\"


def folder_pre_mask2(path1, path2):
    for file in os.listdir(path1):
        if file.endswith(".png"):
            im_op = skimage.io.imread(path1 + "\\" + file)
            image = pre_mask2(im_op)
            image = unsharp_mask(image, radius=5, amount=2)
            image_int8 = skimage.img_as_ubyte(image)
            skimage.io.imsave(path2 + file, image_int8)
# ------------------------



def folder_pre_mask2_sh2(path1, path2):
    for file in os.listdir(path1):
        if file.endswith(".png"):
            im_op = skimage.io.imread(path1 + "\\" + file)
            image = pre_mask2(im_op)
            image = unsharp_mask(image, radius=2, amount=2)
            image_int8 = skimage.img_as_ubyte(image)
            skimage.io.imsave(path2 + file, image_int8)

# folder_pre_mask2(path1, path2)
# PREPROCESSING MASK1
# Image will be
# 1) 874px in width and height*874/width
# 2) noise reduction
def pre_mask1(image):
    height, width = image.shape[:2]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # denoising
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)  #

    if width < height * 2:
        image = skimage.transform.resize(opening, (int(height * 874 / width), int(874)))
    else:
        image = skimage.transform.resize(opening, (int(874), int(width * 874 / height)))
    image = skimage.img_as_ubyte(image)
    return image
# ------------------------


# POSTPROCESSING
# def process_image(image, contrast=1):
#     # image = Image.open(filepath)
#     image_resized = np.resize(image, (874, 614))
#     print(image_resized)
#     # image = image.resize((874, 614))
#     image = Image.fromarray(image_resized, mode='RGB')
#     enhancer = ImageEnhance.Contrast(image)
#     image = enhancer.enhance(contrast)
#     image.show()
#     return image


def resize_folder(folderpath):
    for file in os.listdir(path=folderpath):
        if file.endswith('.png'):
            image = Image.open(folderpath + '\\' + file)
            width, height = image.size
            image = image.resize((874, int(height / width * 874)))
            image.save("C:\\Users\\kuanyshov.a\\Documents\\MaskRCNN\\Model_creating_process\\data\\images\\SECTIONS\\new_model_train_resized" + '\\' + file, quality=100, dpi=(300,300))


# resize_folder("C:\\Users\\kuanyshov.a\\Documents\\MaskRCNN\\Model_creating_process\\data\\images\\SECTIONS\\new_model_train")



def process_image(image):
    image_resized = tr.resize(image, (614, 874),
                           anti_aliasing=True)
    image = exposure.rescale_intensity(image_resized)
    return image


if __name__ == "__main__":
    folderpath = "C:\\Users\\kuanyshov.a\\Documents\\MaskRCNN\\Model_creating_process\\data\\images\\STAMP\\data\\Blueprint_1.1\\after_pre_mask\\train\\stamp"
    for i in os.listdir(folderpath):
        if i.endswith("15.png") or i.endswith("32.png") or i.endswith("38.png") or i.endswith("61.png") or i.endswith("62.png") or i.endswith("63.png") or i.endswith("64.png") or i.endswith("65.png") or i.endswith("66.png"):
            image = skimage.io.imread(folderpath + "\\" + i)
            height, width = image.shape[:2]

            image = skimage.transform.resize(image, (int(874), int(width * 874 / height)))

            skimage.io.imsave(folderpath + "\\" + i, image)

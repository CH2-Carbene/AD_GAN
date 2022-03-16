from random import random
import nibabel as nib
import numpy as np
from GAN.prep import random_jitter,normalize, random_select

def load_img(img_file):
    '''Load one imgs & normalization to float64 tensor'''
    img=nib.load(img_file).get_fdata(caching='unchanged')
    # brain= img[img!=0]
    # brain_norm=(brain-np.mean(brain))/np.std(brain)
    # img_norm=np.zeros_like(img)
    # img_norm[img!=0]=brain_norm
    return img

def load_pair(img_file_dir,input_img_name="T1.nii.gz",real_img_name="FA.nii.gz"):
    '''Load paired img from dir'''
    if type(img_file_dir)!=str:
        img_file_dir=img_file_dir.decode()
    input_img_fullname=f"{img_file_dir}/{input_img_name}"
    real_img_fullname=f"{img_file_dir}/{real_img_name}"
    input_img,real_img=load_img(input_img_fullname),load_img(real_img_fullname)
    if input_img_name=="T1.nii.gz":input_img[real_img==0]=0
    else:real_img[input_img==0]=0
    return input_img,real_img

def load_image_train(image_dir,direct="T1_to_FA"):
    input_img_name,real_img_name="T1.nii.gz","FA.nii.gz"
    if direct!="T1_to_FA":
        input_img_name,real_img_name="FA.nii.gz","T1.nii.gz"
    input_image, real_image = load_pair(image_dir,input_img_name,real_img_name)
    input_image, real_image = random_jitter(input_image,real_image)
    input_image, real_image = np.tanh(input_image), np.tanh(real_image)

    return input_image.astype("float32"), real_image.astype("float32")

def load_image_test(image_dir,direct="T1_to_FA"):
    input_img_name,real_img_name="T1.nii.gz","FA.nii.gz"
    if direct!="T1_to_FA":
        input_img_name,real_img_name="FA.nii.gz","T1.nii.gz"
    input_image, real_image = load_pair(image_dir,input_img_name,real_img_name)
    input_image, real_image = random_select(input_image, real_image)
    input_image, real_image = np.tanh(input_image), np.tanh(real_image)
    return input_image.astype("float32"), real_image.astype("float32")
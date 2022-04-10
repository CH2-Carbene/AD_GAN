import nibabel as nib
import numpy as np
from .prep import random_jitter,normalize, random_select
import tensorflow as tf

def load_np_data(filename,argu=False):
    st=(17, 26, 5)
    ed=(-18, -22, -30)
    if type(filename)!=str:
        filename=filename.decode()
    data=np.load(filename,mmap_mode="r")
    
    # if argu:
        # x,y=random_jitter(data)
    # else:
    data=data[:,st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
    if argu:
        x,y=random_jitter(data,only_select=True)
    else:
        x,y=random_jitter(data,only_select=False)
    # x,y=np.pad(x,((16,16),(0,0),(16,16))),np.pad(y,((16,16),(0,0),(16,16)))
    mask=(x!=0)&(y!=0)
    x=x*mask;y=y*mask
    # if GAN_DIRECT=="FA-T1":x,y=y,x
    return tf.convert_to_tensor(x[...,tf.newaxis]),tf.convert_to_tensor(y[...,tf.newaxis])

def load_img(img_file):
    '''Load one imgs & normalization to float64 tensor'''
    img=nib.load(img_file).get_fdata(caching='unchanged')
    # brain= img[img!=0]
    # brain_norm=(brain-np.mean(brain))/np.std(brain)
    # img_norm=np.zeros_like(img)
    # img_norm[img!=0]=brain_norm
    return img

def load_subject(img_file_dir,T1_name="T1.nii.gz",others_name=["FA.nii.gz"]):
    '''Load paired img from dir'''
    st=(17, 26, 5)
    ed=(-18, -22, -30)
    T1_img=load_img(f"{img_file_dir}/{T1_name}")
    others_img=[load_img(f"{img_file_dir}/{oi_name}")for oi_name in others_name]
    for oi in others_img:
        oi[T1_img==0]=0
    T1_img=normalize(T1_img).astype("float32")[st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
    for i in range(len(others_img)):
        others_img[i]=normalize(others_img[i]).astype("float32")[st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
    res={"T1":T1_img}
    for i in range(len(others_name)):
        res[others_name[i][:others_name[i].find('.')]]=others_img[i]
    # if input_img_name=="T1.nii.gz":input_img[real_img==0]=0
    # else:real_img[input_img==0]=0
    return res

def load_pair(img_file_dir,input_img_name="T1.nii.gz",real_img_name="FA.nii.gz"):
    '''Load paired img from dir'''
    if type(img_file_dir)!=str:
        img_file_dir=img_file_dir.decode()
    input_img_fullname=f"{img_file_dir}/{input_img_name}"
    real_img_fullname=f"{img_file_dir}/{real_img_name}"
    input_img,real_img=load_img(input_img_fullname),load_img(real_img_fullname)
    input_img[real_img==0]=0
    real_img[input_img==0]=0
    # if input_img_name=="T1.nii.gz":input_img[real_img==0]=0
    # else:real_img[input_img==0]=0
    input_img, real_img=normalize(input_img), normalize(real_img)
    return input_img,real_img

def load_image_train(image_dir,direct="T1_to_FA"):
    input_img_name,real_img_name="T1.nii.gz","FA.nii.gz"
    if direct!="T1_to_FA":
        input_img_name,real_img_name="FA.nii.gz","T1.nii.gz"
    input_image, real_image = load_pair(image_dir,input_img_name,real_img_name)
    # input_image, real_image = random_jitter(input_image,real_image)
    # input_image, real_image = random_select(input_image, real_image)
    input_image, real_image = np.tanh(input_image), np.tanh(real_image)
    return input_image.astype("float32"), real_image.astype("float32")

def load_image_test(image_dir,direct="T1_to_FA"):
    input_img_name,real_img_name="T1.nii.gz","FA.nii.gz"
    if direct!="T1_to_FA":
        input_img_name,real_img_name="FA.nii.gz","T1.nii.gz"
    input_image, real_image = load_pair(image_dir,input_img_name,real_img_name)
    # input_image, real_image = random_select(input_image, real_image)
    input_image, real_image = np.tanh(input_image), np.tanh(real_image)
    return input_image.astype("float32"), real_image.astype("float32")
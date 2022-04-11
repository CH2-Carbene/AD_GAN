from units.prep import random_jitter
import numpy as np
import tensorflow as tf
import nibabel as nib
import numpy as np
from .prep import normalize
import tensorflow as tf

def load_np_data(filename,load_mods=["T1","FA"],argu=False):
    # st=(17, 26, 5)
    # ed=(-18, -22, -30)
    if type(filename)!=str:
        filename=filename.decode()

    data=np.load(filename,mmap_mode="r")
    imgs=np.array([data[md.decode()if type(md)!=str else md]for md in load_mods])
    # data=data[:,st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
    if argu:
        imgs=random_jitter(imgs,only_select=True)
    else:
        imgs=random_jitter(imgs,only_select=False)
    # x,y=np.pad(x,((16,16),(0,0),(16,16))),np.pad(y,((16,16),(0,0),(16,16)))
    # mask=(x!=0)&(y!=0)
    # x=x*mask;y=y*mask
    # if GAN_DIRECT=="FA-T1":x,y=y,x
    return [tf.convert_to_tensor(img)[...,tf.newaxis] for img in imgs]


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

import os
import tensorflow as tf
import numpy as np
import multiprocessing
from dataloader import load_pair,load_image_train,load_image_test
from prep import random_jitter
from model import Generator,Discriminator
from loss import get_gen_loss,get_disc_loss

DATAPATH = "../datasets/brainmap/paired"
NEWPATH="../datasets/brainmap/npdata"
# if DEBUG:data=data[:10]
# show(data)
def load_prep_image(image_dir,direct="T1_to_FA"):
    input_img_name,real_img_name="T1.nii.gz","FA.nii.gz"
    if direct!="T1_to_FA":
        input_img_name,real_img_name="FA.nii.gz","T1.nii.gz"
    input_image, real_image = load_pair(image_dir,input_img_name,real_img_name)
    input_image, real_image = np.tanh(input_image), np.tanh(real_image)
    # input_image, real_image = random_jitter(input_image,real_image)
    return input_image.astype("float32"), real_image.astype("float32")

def prep_data(data):
    ds=load_prep_image(f"{DATAPATH}/{data}")
    np.save(f"{NEWPATH}/{data}",np.array(ds))

if __name__ == '__main__':
    data=[f"{imgdir}"for imgdir in os.listdir(DATAPATH)]
    data=[data[0],data[1]]
    os.makedirs(NEWPATH,exist_ok=True)

    pn=40
    mulpool_flt = multiprocessing.Pool(processes=pn)
    fail_set=set()
    for fname in data:
        # prep_data(fname)
        mulpool_flt.apply_async(prep_data,args=(fname,),error_callback=lambda e:fail_set.add(e))
    mulpool_flt.close()
    mulpool_flt.join()

    print(f"fail_set: {fail_set}")

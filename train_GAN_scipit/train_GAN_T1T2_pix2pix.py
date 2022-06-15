# %%
import tensorflow as tf
# print('import tf', tf.__version__)

import os,pickle
import time
import datetime
from imp import reload
from matplotlib import pyplot as plt
import numpy as np
import sys
from units.globals import DEBUG
from units.globals import DEBUG,NEWPATH,N_CPU
import argparse

from units.base import visualize,generate_images,show
from units.dataloader import load_np_data
from sklearn.model_selection import train_test_split

# %%
parser=argparse.ArgumentParser(description="cyc_GAN_model")
parser.add_argument("-l","--lamda",type=float,default=10, help="set cycle-consistency lamda value")
parser.add_argument("-b","--batch_size",type=int,default=1, help="set batch size")
parser.add_argument("-e","--epoches",type=int,default=200, help="set epoches")
parser.add_argument("-m","--model_path",default=None, help="set pretrained model path")
parser.add_argument("-a","--argument",type=bool,default=False, help="whether use argument")
args = parser.parse_args()

LAMDA=args.lamda
BATCH_SIZE=args.batch_size
EPOCHES=args.epoches
MODEL_PATH=args.model_path
ARGU=args.argument


# %%
# reload(GAN)
load_mods=["T1","T2"]
NEWPATH="datasets/brainmap/npdata_t2"
data=[f"{NEWPATH}/{img}"for img in os.listdir(NEWPATH)]

demo=np.load(data[3])
t1,fa=demo[load_mods[0]],demo[load_mods[1]]
# t1[fa==0]=0
visualize([t1,fa],save_path="demo/paired.png")

# %%
from units.dataloader import load_np_data

t1_arg,fa_arg=load_np_data(data[0],load_mods,ARGU)
# print(t1_arg.shape)
# t1_arg,fa_arg=load_image_train(t1_arg,fa_arg)#,[Rotation3D(max_rate=np.pi/2)])
visualize([t1_arg[...,0],fa_arg[...,0]])
np.save("demo/t1_arg",t1_arg)
np.save("demo/fa_arg",fa_arg)

# %%
data=[f"{NEWPATH}/{img}"for img in os.listdir(NEWPATH)]
train_val,test=train_test_split(
    data,test_size=0.1,random_state=1919810
)
train,val=train_test_split(
    train_val,test_size=0.1,random_state=114514
)
show(f"Train len: {len(train)}")
show(f"Val len: {len(val)}")
show(f"Test len: {len(test)}")

# %%
# The facade training set consist of 400 images

BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
# st_range=np.array((227, 272, 227))-np.array((128,128,128))

train_load=lambda filename:tf.numpy_function(func=load_np_data,inp=[filename,load_mods,ARGU],Tout=(tf.float32,tf.float32))
test_load=lambda filename:tf.numpy_function(func=load_np_data,inp=[filename,load_mods,False],Tout=(tf.float32,tf.float32))

def get_train_ds(train):
    # train_dataset=[]
    # for t in tqdm(train):
        # train_dataset.append(load_image_train(t))
    # train_dataset=np.array(train_dataset)
    # train_dataset = list(map(load_image_train,train))
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train)
    # print(train_dataset)
    # train_dataset=load_image_train(train)
    train_dataset = train_dataset.map(map_func=train_load,num_parallel_calls=N_CPU)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE,seed=114514)
    train_dataset = train_dataset.batch(BATCH_SIZE,num_parallel_calls=N_CPU)
    return train_dataset
# train_dataset=train_dataset.map(lambda x:tf.numpy_function(func=upper_case_fn,inp=[x],Tout=(tf.float64,tf.float64)))

def get_test_ds(test):
    # test_dataset=[]
    # for i in range(8):test_dataset+=[load_image_test(test_dir)for test_dir in test]
    # iplist,relist=[],[]
    # for input,real in test_dataset:
    #     iplist.append(input)
    #     relist.append(real)
    test_dataset = tf.data.Dataset.from_tensor_slices(test)
    test_dataset = test_dataset.map(map_func=test_load,num_parallel_calls=N_CPU)
    # test_dataset = test_dataset.map(lambda x:tf.numpy_function(func=load_image_test,inp=[x],Tout=(tf.float32,tf.float32)),num_parallel_calls=16,deterministic=False)
    # test_dataset = test_dataset.map(lambda x:tf.numpy_function(func=load_image_test,inp=[x],Tout=(tf.float32,tf.float32)),num_parallel_calls=tf.data.AUTOTUNE,deterministic=False)
    test_dataset = test_dataset.batch(BATCH_SIZE,num_parallel_calls=N_CPU)
    return test_dataset

train_ds,val_ds,test_ds=get_train_ds(train),get_test_ds(val),get_test_ds(test)

# %%
from GAN.model import Pix2pix
from units.base import generate_images,visualize

tip,fip=t1_arg,fa_arg
p2pgan=Pix2pix(tip.shape,example_data=[tip,fip])
G,D=p2pgan.G,p2pgan.D
if MODEL_PATH is not None:
    p2pgan.load_model(MODEL_PATH)
p2pgan.generate_images([tip[...,0],fip[...,0]])

T2_metric=p2pgan.eval_result(test_ds)
print("T2_metric:",T2_metric)
# generate_images(G,tip[...,0],fip[...,0])
# print(p2pgan.test_step(tip[tf.newaxis,...],fip[tf.newaxis,...]))

# %%
tot_step=len(train_ds)*200
h=p2pgan.train(train_ds,val_ds,steps=tot_step)
with open(f"{p2pgan.log_dir}/training_log.pic","wb") as f:
    pickle.dump(h,f)
from units.base import show_process
show_process(h["train"],labels=p2pgan.outputs ,save_path=f"{p2pgan.log_dir}/train_process")
show_process(h["valid"],labels=p2pgan.outputs, save_path=f"{p2pgan.log_dir}/valid_process")

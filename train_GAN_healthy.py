# %%
from units.base import cyc_generate_images, generate_images, visualize
from units.base import show_process
from pet_cycgan import Cycgan_pet
import tensorflow as tf
# print('import tf', tf.__version__)

import os
import pickle
from imp import reload
from matplotlib import pyplot as plt
import numpy as np
from units.globals import DEBUG, NEWPATH, N_CPU
import argparse

from units.base import visualize, generate_images, show
from units.dataloader import load_np_data
from sklearn.model_selection import train_test_split

# %%
parser = argparse.ArgumentParser(description="cyc_GAN_model")
parser.add_argument("-l", "--lamda", type=float, default=10,
                    help="set cycle-consistency lamda value")
parser.add_argument("-b", "--batch_size", type=int,
                    default=1, help="set batch size")
parser.add_argument("-e", "--epoches", type=int,
                    default=200, help="set epoches")
parser.add_argument("-m", "--model_path", default=None,
                    help="set pretrained model path")
parser.add_argument("-a", "--argument", type=bool,
                    default=False, help="whether use argument")
args = parser.parse_args()

LAMDA = args.lamda
BATCH_SIZE = args.batch_size
EPOCHES = args.epoches
MODEL_PATH = args.model_path
ARGU = args.argument
patch_shape=(128,128,128)

# %%
# reload(GAN)
load_mods = ["T1", "FA"]
ROOT="/public_bme/data/gujch/brainmap"

# ROOT=r"C:\Users\CH2\Projects\AD_GAN\datasets\brainmap"

NEWPATH = ROOT+"/npdata"
HEALCSV = ROOT+"/healthy.csv"
import pandas as pd
hid=list(pd.read_csv(HEALCSV)["PID"])
hid=[p.replace(' ','') for p in hid]
# %%
def checkin(x):
    for p in hid:
        if p in x:return True
    return False
data = [f"{NEWPATH}/{img}"for img in os.listdir(NEWPATH)]
data=list(filter(checkin,data))
print("Total:",len(data))
train, test = train_test_split(
    data, test_size=0.1, random_state=1919810
)
val=test
# train, val = train_test_split(
    # train_val, test_size=0.1, random_state=114514
# )
show(f"Train len: {len(train)}")
show(f"Val len: {len(val)}")
show(f"Test len: {len(test)}")

# %%
# The facade training set consist of 400 images

t1_arg, fa_arg = load_np_data(val[0], load_mods, ARGU)
# print(t1_arg.shape)
# t1_arg,fa_arg=load_image_train(t1_arg,fa_arg)#,[Rotation3D(max_rate=np.pi/2)])
visualize([t1_arg[..., 0], fa_arg[..., 0]])
np.save("demo/t1_arg", t1_arg)
np.save("demo/fa_arg", fa_arg)

BUFFER_SIZE = 300
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
# st_range=np.array((227, 272, 227))-np.array((128,128,128))


def train_load(filename): return tf.numpy_function(func=load_np_data, inp=[
    filename, load_mods, ARGU,patch_shape], Tout=(tf.float32, tf.float32))


def test_load(filename): return tf.numpy_function(func=load_np_data, inp=[
    filename, load_mods, False,patch_shape], Tout=(tf.float32, tf.float32))


def get_train_ds(train):
    # train_dataset=[]
    # for t in tqdm(train):
    # train_dataset.append(load_image_train(t))
    # train_dataset=np.array(train_dataset)
    # train_dataset = list(map(load_image_train,train))

    train_dataset = tf.data.Dataset.from_tensor_slices(train)
    # print(train_dataset)
    # train_dataset=load_image_train(train)
    train_dataset = train_dataset.map(
        map_func=train_load, num_parallel_calls=N_CPU)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE, seed=114514)
    train_dataset = train_dataset.batch(BATCH_SIZE, num_parallel_calls=N_CPU)
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
    test_dataset = test_dataset.map(
        map_func=test_load, num_parallel_calls=N_CPU)
    # test_dataset = test_dataset.map(lambda x:tf.numpy_function(func=load_image_test,inp=[x],Tout=(tf.float32,tf.float32)),num_parallel_calls=16,deterministic=False)
    # test_dataset = test_dataset.map(lambda x:tf.numpy_function(func=load_image_test,inp=[x],Tout=(tf.float32,tf.float32)),num_parallel_calls=tf.data.AUTOTUNE,deterministic=False)
    test_dataset = test_dataset.batch(BATCH_SIZE, num_parallel_calls=N_CPU)
    return test_dataset


train_ds, val_ds, test_ds = get_train_ds(
    train), get_test_ds(val), get_test_ds(test)

# %%

tip, fip = t1_arg, fa_arg

print(tip.shape)
cycgan = Cycgan_pet(tip.shape, LAMDA, example_data=[
                    tip, fip], modality="T1-FA")
G1, G2, DA, DB = cycgan.G1, cycgan.G2, cycgan.DA, cycgan.DB
cyc_generate_images(G1, G2, tip[..., 0], fip[..., 0])
print(cycgan.test_step(tip[tf.newaxis, ...], fip[tf.newaxis, ...]))

# %%

tot_step = len(train_ds)*200
h = cycgan.train(train_ds, val_ds, steps=tot_step)
with open(f"{cycgan.log_dir}/training_log.pic", "wb") as f:
    pickle.dump(h, f)
show_process(h["train"], labels=cycgan.outputs,
             save_path=f"{cycgan.log_dir}/train_process")
show_process(h["valid"], labels=cycgan.outputs,
             save_path=f"{cycgan.log_dir}/valid_process")

metric = cycgan.eval_result(val_ds)
print("Fake_metric:", metric)

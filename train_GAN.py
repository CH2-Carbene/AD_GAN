# %%
import tensorflow as tf

import os
import time
import datetime
from imp import reload
from matplotlib import pyplot as plt
import numpy as np
from sys import stdout
from units.globals import DEBUG

# %%
### units for data preprocession
# reload(units.base)
from GAN.dataloader import load_pair
from units.base import visualize,generate_images
# DEBUG=True
print("Start processing...")
t1,fa=load_pair(r"./datasets/brainmap/paired/9")
# t1[fa==0]=0
visualize([t1,fa],save_path="demo/paired.png")
# visualize(fa)

# %%
### visualize argument
# reload(GAN.prep)
from GAN.prep import random_jitter

# for i in range(1):
t1_arg,fa_arg=t1,fa#Patch_extration()(t1,fa)
t1_arg,fa_arg=random_jitter(t1_arg,fa_arg)#,[Rotation3D(max_rate=np.pi/2)])
visualize([t1_arg,fa_arg],save_path="demo/paired_arg.png")
# visualize(fa_arg)
np.save("demo/t1_arg",t1_arg)
np.save("demo/fa_arg",fa_arg)

# %%
### train_test_split
# DEBUG=True
DATAPATH = "./datasets/brainmap/paired"
data=[f"{DATAPATH}/{imgdir}"for imgdir in os.listdir(DATAPATH)]
if DEBUG:data=data[:10]
# print(data)
from sklearn.model_selection import train_test_split
train_val,test=train_test_split(
    data,test_size=0.1,random_state=1919810
)
train,val=train_test_split(
    train_val,test_size=0.1,random_state=114514
)
# train
print(f"Train len: {len(train)}")
print(f"Val len: {len(val)}")
print(f"Test len: {len(test)}")

# %%
# The facade training set consist of 400 images
BUFFER_SIZE = 16
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 4
import GAN
reload(GAN)
from GAN.dataloader import load_image_train,load_image_test
from tqdm import tqdm
# print(f"train_dataset: {len(train_dataset)}")
def get_train_ds(train):
    # train_dataset=[]
    # for t in tqdm(train):
        # train_dataset.append(load_image_train(t))
    # train_dataset=np.array(train_dataset)
    # train_dataset = list(map(load_image_train,train))
    train_dataset = tf.data.Dataset.from_tensor_slices(train)
    # train_dataset=load_image_train(train)
    train_dataset = train_dataset.map(lambda x:tf.numpy_function(func=load_image_train,inp=[x],Tout=(tf.float32,tf.float32)),num_parallel_calls=tf.data.AUTOTUNE,deterministic=False)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE,seed=114514)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    return train_dataset
# train_dataset=train_dataset.map(lambda x:tf.numpy_function(func=upper_case_fn,inp=[x],Tout=(tf.float64,tf.float64)))

def get_test_ds(test):
    test_dataset=[]
    for i in range(8):test_dataset+=[load_image_test(test_dir)for test_dir in test]
    iplist,relist=[],[]
    for input,real in test_dataset:
        iplist.append(input)
        relist.append(real)
    test_dataset = tf.data.Dataset.from_tensor_slices((iplist,relist))
    # test_dataset = test_dataset.map(lambda x:tf.numpy_function(func=load_image_test,inp=[x],Tout=(tf.float32,tf.float32)),num_parallel_calls=tf.data.AUTOTUNE,deterministic=False)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    return test_dataset

train_ds,val_ds,test_ds=get_train_ds(train),get_test_ds(val),get_test_ds(test)


# %%
gen_oper = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_oper = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

from GAN.model import Generator,Discriminator
# reload(GAN.model)
generator = Generator()
discriminator=Discriminator()
from GAN.loss import get_gen_loss,get_disc_loss
# reload(GAN.loss)


# %%
log_dir="logs/"
this_log_dir=log_dir + "GAN_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(this_log_dir)
path=f"{this_log_dir}/T1_FA"
checkpoint_dir = f"{this_log_dir}/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_oper,
                                 discriminator_optimizer=disc_oper,
                                 generator=generator,
                                 discriminator=discriminator)


# %%
ALPHA=5
G,D=generator,discriminator
@tf.function
def train_step(img, tar, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        tar_fake = G(img, training=True)

        disc_real_output = D([img, tar], training=True)
        disc_fake_output = D([img, tar_fake], training=True)

        gen_loss, dice_loss, gan_disc_loss = get_gen_loss(
            tar,tar_fake,disc_fake_output, ALPHA)
        disc_loss = get_disc_loss(disc_real_output, disc_fake_output)

    gen_grad = gen_tape.gradient(
        gen_loss, G.trainable_variables)
    disc_grad = disc_tape.gradient(
        disc_loss, D.trainable_variables)

    gen_oper.apply_gradients(
        zip(gen_grad, G.trainable_variables))
    disc_oper.apply_gradients(
        zip(disc_grad, D.trainable_variables))

    
    with summary_writer.as_default():
        tf.summary.scalar('gen_loss', gen_loss, step)
        tf.summary.scalar('dice_loss', dice_loss, step)
        tf.summary.scalar('gan_disc_loss', gan_disc_loss,step)
        tf.summary.scalar('disc_loss', disc_loss,step)
    return gen_loss, dice_loss, gan_disc_loss, disc_loss

# %%
@tf.function
def test_step(img, tar):
    tar_fake = G(img, training=False)

    disc_real_output = D([img, tar], training=False)
    disc_fake_output = D([img, tar_fake], training=False)
    disc_loss = get_disc_loss(disc_real_output, disc_fake_output)

    gen_loss, dice_loss, gan_disc_loss = get_gen_loss(tar, tar_fake, disc_fake_output, ALPHA)
        
    return gen_loss, dice_loss, gan_disc_loss, disc_loss

# %%
val_time=100

def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    history = {'train': [], 'valid': []}
    prev_loss = np.inf

    train_losses=[tf.keras.metrics.Mean() for i in range(4)]
    test_losses=[tf.keras.metrics.Mean() for i in range(4)]

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():

        start = time.time()
        print('Epoch {}/{}'.format(step+1,steps))

        # if (step+1) % 1 == 0:
            # display.clear_output(wait=True)
            # generate_images(generator, example_input, example_target)

        train_step_loss=train_step(input_image, target, step)
        for meti,li in zip(train_losses,train_step_loss):meti.update_state(li)
        gen_loss, dice_loss, gan_disc_loss, disc_loss=[x.result() for x in train_losses]

        stdout.write(f'\rStep: {step+1}/{steps} - loss: {gen_loss:.6f} - dice_loss: {dice_loss:.6f} - gan_disc_loss: {gan_disc_loss:.6f} - disc_loss: {disc_loss:.6f}')
        
        print(f'Time taken for 1 steps: {time.time()-start:.2f} sec\n')
        stdout.flush()
        
        if (step+1) % val_time==0:
            
            for step, (input_image, target) in test_ds.enumerate():
                test_step_loss=test_step(input_image, target)
                for meti,li in zip(test_losses,test_step_loss):meti.update_state(li)
            
            gen_loss_val, dice_loss_val, gan_disc_loss_val, disc_loss_val=[x.result() for x in test_losses]
            stdout.write(f'\rVal_step: {(step+1)//val_time}/{steps//val_time} - val_loss: {gen_loss_val:.6f} - val_dice_loss: {dice_loss_val:.6f} - val_gan_disc_loss: {gan_disc_loss_val:.6f} - val_disc_loss: {disc_loss_val:.6f}')


            save_path=f"{path}/step_{step:03d}"
            os.makedirs(save_path,exist_ok=True)
            
            generate_images(G,example_input, example_target,save_path=f"{save_path}/show.png")
            G.save_weights(f"{save_path}/G.h5") 
            D.save_weights(f"{save_path}/D.h5") 

            if gen_loss_val < prev_loss:    
                G.save_weights(f"{path}/Generator.h5") 
                D.save_weights(f"{path}/Generator.h5") 
                print(f"Validation loss decresaed from {prev_loss:.4f} to {gen_loss_val:.4f}. Models' weights are now saved.")
                prev_loss=gen_loss_val
            else:
                print(f"Validation loss did not decrese from {prev_loss:.4f} to {gen_loss_val:.4f}.")

            history['train'].append([x.result() for x in train_losses])
            history['valid'].append([x.result() for x in test_losses])
            for x in train_losses:x.reset_states()
            for x in test_losses:x.reset_states()
            checkpoint.save(file_prefix=checkpoint_prefix)
    return history

# %%
h=fit(train_ds,val_ds,steps=10)
print(h)
# # %%
# h

# # %%
# #try tensor.map
# train_dataset = tf.data.Dataset.from_tensor_slices(train)
# # d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
# def upper_case_fn(t: np.ndarray):
#     return t.decode('utf-8')+"1",t.decode('utf-8')+"2"
# # d = d.map(lambda x: tf.numpy_function(func=upper_case_fn,
#         #   inp=[x], Tout=tf.string))
# # list(d.as_numpy_iterator())
# train_dataset=train_dataset.map(lambda x:tf.numpy_function(func=upper_case_fn,inp=[x],Tout=(tf.float64,tf.float64)))
# # train_dataset=train_dataset.map(lambda x:tf.strings.as_string(x)+"2")
# list(train_dataset.as_numpy_iterator())
# # len(train_dataset.map(np.array))
# # for i in train_dataset:
#     # print()#.numpy())


# # %%
# from numpy.random import randint
# d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
# def upper_case_fn(t: np.ndarray):
#   return (t.decode('utf-8')+str(randint(10))).upper()
# d = d.map(lambda x: tf.numpy_function(func=upper_case_fn,
#           inp=[x], Tout=tf.string),num_parallel_calls=tf.data.AUTOTUNE,deterministic=False)
# list(d.repeat().take(10).as_numpy_iterator())
# # list(d.as_numpy_iterator())

# # %%
# class A:
#     def __init__(self):
#         self.t=10
#     def exec(self,b):
#         self.t+=b
# a=[A()for i in range(4)]
# b=[i for i in range(4)]
# map(lambda x,y:x.exec(y),a,b)
# a[2].t

# # %%




# %%
import tensorflow as tf
# print('import tf', tf.__version__)

import os,pickle
import time
import datetime
from imp import reload
from matplotlib import pyplot as plt
import numpy as np
from sys import stdout
from units.globals import DEBUG

print('import module')
os.system('pwd')

# %%
### units for data preprocession
# reload(units.base)
import GAN
from GAN.dataloader import load_pair,load_image_train,load_image_test
from GAN.prep import random_jitter
from GAN.model import Generator,Discriminator
from GAN.loss import get_gen_loss,get_disc_loss

from units.base import visualize,generate_images,show
from sklearn.model_selection import train_test_split
# DEBUG=True



if __name__ == '__main__':
    show("Start processing...")

    # visualize(fa)
    # %%
    ### visualize argument
    # reload(GAN.prep)
    

    # for i in range(1):
    # t1_arg,fa_arg=t1,fa#Patch_extration()(t1,fa)
    # t1_arg,fa_arg=random_jitter(t1_arg,fa_arg)#,[Rotation3D(max_rate=np.pi/2)])
    # visualize([t1_arg,fa_arg],save_path="demo/paired_arg.png")
    # visualize(fa_arg)
    # np.save("demo/t1_arg",t1_arg)
    # np.save("demo/fa_arg",fa_arg)

    # %%
    ### train_test_split
    # DEBUG=True
    # DATAPATH = "./datasets/brainmap/paired"
    # data=[f"{DATAPATH}/{imgdir}"for imgdir in os.listdir(DATAPATH)]
    NEWPATH="datasets/brainmap/npdata"
    data=[f"{NEWPATH}/{img}"for img in os.listdir(NEWPATH)]
    demo=np.load(data[0])
    t1,fa=demo[0],demo[1]
    # t1[fa==0]=0
    visualize([t1,fa],save_path="demo/paired.png")
    # if DEBUG:data=data[:10]
    # show(data)
    # if DEBUG:
    train_val,test=train_test_split(
        data,test_size=0.1,random_state=1919810
    )
    train,val=train_test_split(
        train_val,test_size=0.1,random_state=114514
    )
    # else:
    # train,val,test=data,data,data
    # train
    show(f"Train len: {len(train)}")
    show(f"Val len: {len(val)}")
    show(f"Test len: {len(test)}")

    # %%
    # The facade training set consist of 400 images
    BUFFER_SIZE = 400
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 4
    st_range=np.array((227, 272, 227))-np.array((128,128,128))
    def load_np_data(filename):
        if type(filename)!=str:
            filename=filename.decode()
        data=np.load(filename,mmap_mode="r")
            # tf.random.
        st=np.random.randint(st_range)
        ed=st+128
        x,y=data[0,st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]],data[1,st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
        return tf.convert_to_tensor(x),tf.convert_to_tensor(y)
            
    warp_load_np=lambda x:tf.numpy_function(func=load_np_data,inp=[x],Tout=(tf.float32,tf.float32))

    def get_train_ds(train):
        # train_dataset=[]
        # for t in tqdm(train):
            # train_dataset.append(load_image_train(t))
        # train_dataset=np.array(train_dataset)
        # train_dataset = list(map(load_image_train,train))
        
        train_dataset = tf.data.Dataset.from_tensor_slices(train)
        # print(train_dataset)
        # train_dataset=load_image_train(train)
        train_dataset = train_dataset.map(map_func=warp_load_np,num_parallel_calls=8)
        train_dataset = train_dataset.shuffle(BUFFER_SIZE,seed=114514)
        train_dataset = train_dataset.batch(BATCH_SIZE,num_parallel_calls=8)
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
        test_dataset = test_dataset.map(map_func=warp_load_np,num_parallel_calls=8)
        # test_dataset = test_dataset.map(lambda x:tf.numpy_function(func=load_image_test,inp=[x],Tout=(tf.float32,tf.float32)),num_parallel_calls=16,deterministic=False)
        # test_dataset = test_dataset.map(lambda x:tf.numpy_function(func=load_image_test,inp=[x],Tout=(tf.float32,tf.float32)),num_parallel_calls=tf.data.AUTOTUNE,deterministic=False)
        test_dataset = test_dataset.batch(BATCH_SIZE,num_parallel_calls=8)
        return test_dataset

    train_ds,val_ds,test_ds=get_train_ds(train),get_test_ds(val),get_test_ds(test)


    # %%
    gen_oper = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_oper = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    
    # reload(GAN.model)
    generator = Generator()
    discriminator=Discriminator()
    
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
    val_time=len(train_ds)//BATCH_SIZE*5
    tot_step=len(train_ds)*500

    def fit(train_ds, test_ds, steps):
        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()
        history = {'train': [], 'valid': []}
        prev_loss = np.inf

        train_losses=[tf.keras.metrics.Mean() for i in range(4)]
        test_losses=[tf.keras.metrics.Mean() for i in range(4)]

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():

            start = time.time()
            show(f'Epoch {step+1}/{steps}')

            # if (step+1) % 1 == 0:
                # display.clear_output(wait=True)
                # generate_images(generator, example_input, example_target)

            train_step_loss=train_step(input_image, target, step)
            for meti,li in zip(train_losses,train_step_loss):meti.update_state(li)
            gen_loss, dice_loss, gan_disc_loss, disc_loss=[x.result() for x in train_losses]

            show(f'\rStep: {step+1}/{steps} - loss: {gen_loss:.6f} - dice_loss: {dice_loss:.6f} - gan_disc_loss: {gan_disc_loss:.6f} - disc_loss: {disc_loss:.6f}')
            
            show(f'Time taken for 1 steps: {time.time()-start:.2f} sec\n')
            
            if (step+1) % val_time==0:
                
                for val_step, (input_image, target) in test_ds.enumerate():
                    test_step_loss=test_step(input_image, target)
                    for meti,li in zip(test_losses,test_step_loss):meti.update_state(li)
                
                gen_loss_val, dice_loss_val, gan_disc_loss_val, disc_loss_val=[x.result() for x in test_losses]
                show(f'\rVal_step: {(step+1)//val_time}/{steps//val_time} - val_loss: {gen_loss_val:.6f} - val_dice_loss: {dice_loss_val:.6f} - val_gan_disc_loss: {gan_disc_loss_val:.6f} - val_disc_loss: {disc_loss_val:.6f}')


                save_path=f"{path}/step_{step+1:03d}"
                os.makedirs(save_path,exist_ok=True)
                
                generate_images(G,example_input, example_target,save_path=f"{save_path}/show.png")
                G.save(f"{save_path}/G.h5") 
                D.save(f"{save_path}/D.h5") 

                if gen_loss_val < prev_loss:    
                    G.save(f"{path}/Generator.h5") 
                    D.save(f"{path}/Discriminator.h5") 
                    show(f"Validation loss decresaed from {prev_loss:.4f} to {gen_loss_val:.4f}. Models' weights are now saved.")
                    prev_loss=gen_loss_val
                else:
                    show(f"Validation loss did not decrese from {prev_loss:.4f} to {gen_loss_val:.4f}.")

                history['train'].append([x.result().numpy() for x in train_losses])
                history['valid'].append([x.result().numpy() for x in test_losses])
                for x in train_losses:x.reset_states()
                for x in test_losses:x.reset_states()
                checkpoint.save(file_prefix=checkpoint_prefix)
        return history

    # %%
    h=fit(train_ds,val_ds,steps=tot_step)
    show(h)
    h_bytes=pickle.dumps(h)
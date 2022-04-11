import time
import datetime
import os
import pickle
import numpy as np
import tensorflow as tf
from units.base import show, generate_images
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate, ZeroPadding3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from .loss import L1_loss, L2_loss
from .layers import G_conv3d, G_deconv3d, Reslayer


def Generator(input_shape):

    layers_to_concatenate = []
    # inputs = Input((192, 224, 192, 1), name='input_image')
    inputs = Input(input_shape, name='input_image')
    nf_start = 8
    depth, resnum = 3, 3
    ks0 = 7
    ks = 3
    x = inputs

    # encoder
    for d in range(depth):
        if d == 0:
            x = G_conv3d(nf_start, ks=ks0, st=1, pad="reflect")(x)
        else:
            x = G_conv3d(nf_start*np.power(2, d), ks=ks, st=2, pad="same")(x)
        layers_to_concatenate.append(x)

    # bottlenek
    for i in range(resnum):
        x = Reslayer(nf_start*np.power(2, depth-1), ks)(x)
        # x = Reslayer(nf_start, ks)(x)

    # decoder
    for d in range(depth-1, -1, -1):
        if d != 0:
            x = G_deconv3d(nf_start*np.power(2, d-1),
                           ks=ks, st=2, pad="same")(x)
        else:
            x = G_conv3d(1, ks=ks0, st=1, pad="reflect")(x)
    outputs = ReLU()(x/2+1)-1
    return Model(inputs=inputs, outputs=outputs, name='Generator')


def Discriminator(input_shape):
    inputs = Input(input_shape, name='input_image')
    targets = Input(input_shape, name='target_image')
    ks = 4
    depth = 3
    nf_start = 16

    x = Concatenate()([inputs, targets])
    for d in range(depth):
        if d == 0:
            x = G_conv3d(nf_start, ks=ks, st=2, pad="same",
                         norm=False, lrelu=0.2)(x)
        else:
            x = G_conv3d(nf_start*np.power(2, d), ks=ks,
                         st=2, pad="same",  lrelu=0.2)(x)

    x = G_conv3d(nf_start*np.power(2, depth), ks=ks,
                 st=1, pad="same", lrelu=0.2)(x)
    outputs = G_conv3d(1, ks=ks, st=2, pad="same",
                       norm=False, do_relu=False)(x)
    return Model(inputs=[inputs,targets], outputs=outputs, name='Discriminator')

# def add(f1,f2):
    # return lambda x:f1(*x)+f2(*x)


def showState(d: dict):
    for x, y in d.items():
        show(f"{x} : {y:6f}")
    show("")


class Pix2pix:
    def __init__(self, input_shape=(128, 128, 128, 1), alpha=5, example_data=None):
        """
        Pix2pix with paired data. Disc for True and fake image, and L2 loss for voxel loss.
        """

        self.alpha = alpha
        self.input_shape = input_shape
        self.calc_vox_loss = lambda fak,tru: L2_loss(fak,tru)
        self.calc_gan_dis_loss = lambda dis_out: L2_loss(dis_out, 1)
        self.calc_gan_loss = lambda vox_loss, dis_loss: self.alpha * \
            vox_loss+dis_loss

        self.calc_dis_loss = lambda fak, tru: (
            L2_loss(fak, 0) + L2_loss(tru, 1)) / 2

        # curr_lr = 0.0002*(200-max(epoch, 100))/100

        self.G = Generator(input_shape)
        self.D = Discriminator(input_shape)

        self.G_op = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.D_op = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.outputs=["G_loss", "vox_loss", "gan_dis_loss", "D_loss"]

        self.applyop = lambda tape, op, model, loss: op.apply_gradients(
            zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

        self.log_dir = "logs/" + \
            f"alpha{self.alpha}" + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = f"{self.log_dir}/Pix2pix"
        self.prev_loss = np.inf
        if example_data is not None:
            x, y = example_data[0], example_data[1]
            x, y = tf.reshape(
                x, input_shape[:-1]), tf.reshape(y, input_shape[:-1])
            self.example = [x, y]
        else:
            self.example = None

    def save_checkpoint(self, step, now_loss):
        G, D = self.G, self.D
        path, prev_loss = self.path, self.prev_loss
        save_path = f"{path}/step_{step+1:03d}"

        os.makedirs(save_path, exist_ok=True)

        if self.example is not None:
            generate_images(G, self.example[0], self.example[1],
                            save_path=f"{save_path}/show.png")
        G.save(f"{save_path}/G.h5")
        D.save(f"{save_path}/D.h5")

        if now_loss < prev_loss:
            G.save(f"{path}/G.h5")
            D.save(f"{path}/D.h5")
            show(
                f"Validation loss decresaed from {prev_loss:.4f} to {now_loss:.4f}. Models' weights are now saved.")
            self.prev_loss = now_loss
        else:
            show(f"Validation loss did not decrese from {prev_loss:.4f} to {now_loss:.4f}.")

    def test(self, test_ds):
        test_losses = [tf.keras.metrics.Mean() for i in range(4)]
        for val_step, (imgA, imgB) in test_ds.enumerate():
            test_step_loss = self.test_step(imgA, imgB)
            for meti, li in zip(test_losses, test_step_loss):
                meti.update_state(li)
        return [x.result() for x in test_losses]

    def test_step(self, imgA, imgB):
        G, D = self.G, self.D
        calc_gan_dis_loss=self.calc_gan_dis_loss
        calc_gan_loss = self.calc_gan_loss
        calc_dis_loss = self.calc_dis_loss
        calc_vox_loss =self.calc_vox_loss

        fakeB = G(imgA, training=False)
        imgB_dis = D([imgA,imgB], training=False)
        fakeB_dis = D([imgA,fakeB], training=False)

        D_loss = calc_dis_loss(fakeB_dis, imgB_dis)
        vox_loss=calc_vox_loss(imgB,fakeB)
        gan_dis_loss=calc_gan_dis_loss(fakeB_dis)
        G_loss = calc_gan_loss(vox_loss, gan_dis_loss)

        return G_loss, vox_loss, gan_dis_loss, D_loss

    def train_step(self, imgA, imgB):
        """
        imgA:inpup
        imgB:target
        """
        G, D = self.G, self.D
        G_op,D_op=self.G_op,self.D_op
        calc_gan_dis_loss=self.calc_gan_dis_loss
        calc_gan_loss = self.calc_gan_loss
        calc_dis_loss = self.calc_dis_loss
        calc_vox_loss =self.calc_vox_loss

        with tf.GradientTape(persistent=True) as tape:
            fakeB = G(imgA, training=False)
            imgB_dis = D([imgA,imgB], training=False)
            fakeB_dis = D([imgA,fakeB], training=False)

            D_loss = calc_dis_loss(fakeB_dis, imgB_dis)
            vox_loss=calc_vox_loss(imgB,fakeB)
            gan_dis_loss=calc_gan_dis_loss(fakeB_dis)
            G_loss = calc_gan_loss(vox_loss, gan_dis_loss)

        G_grad = tape.gradient(G_loss, G.trainable_variables)
        D_grad = tape.gradient(D_loss, D.trainable_variables)

        G_op.apply_gradients(zip(G_grad, G.trainable_variables))
        D_op.apply_gradients(zip(D_grad, D.trainable_variables))

        return G_loss, vox_loss, gan_dis_loss, D_loss

    def train(self, train_ds, val_ds, batch_size=1, epoches: int = None, steps: int = None, val_time=100):

        history = {'train': [], 'valid': []}
        if steps is None:
            steps = len(train_ds)*400//batch_size
        val_freq = steps//val_time
        train_losses = [tf.keras.metrics.Mean() for i in range(4)]

        start = time.time()

        for step, (imgA, imgB) in train_ds.repeat().take(steps).enumerate():

            show(f'Time taken for 1 dataload: {time.time()-start:.2f} sec\n')
            start = time.time()
            show(f'Step {step+1}/{steps}')

            # if (step+1) % 1 == 0:
            # display.clear_output(wait=True)
            # generate_images(generator, example_input, example_target)

            train_step_loss = self.train_step(imgA, imgB)
            for meti, li in zip(train_losses, train_step_loss):
                meti.update_state(li)

            G_loss, vox_loss, gan_dis_loss, D_loss = train_step_loss

            showState({"G_loss": G_loss, "vox_loss": vox_loss,
                       "gan_dis_loss": gan_dis_loss, "D_loss": D_loss})

            show(f'Time taken for 1 steps: {time.time()-start:.2f} sec\n')

            if (step+1) % val_freq == 0 or step == 0:

                show(f"Val_step: {(step+1)//val_freq}/{val_time}")

                show("Train loss:")
                this_train_losses = [x.result() for x in train_losses]
                G_loss, vox_loss, gan_dis_loss, D_loss = this_train_losses
                showState({"G_loss": G_loss, "vox_loss": vox_loss,
                       "gan_dis_loss": gan_dis_loss, "D_loss": D_loss})

                show("Val loss:")
                this_val_losses = self.test(val_ds)
                G_loss, vox_loss, gan_dis_loss, D_loss = this_val_losses
                showState({"G_loss": G_loss, "vox_loss": vox_loss,
                       "gan_dis_loss": gan_dis_loss, "D_loss": D_loss})

                self.save_checkpoint(step, G_loss)

                history['train'].append(this_train_losses)
                history['valid'].append(this_val_losses)
                for x in train_losses:
                    x.reset_states()

            start = time.time()
        with open(f"{self.log_dir}/training_log.pic", "wb") as f:
            pickle.dump(history, f)
        return history


if __name__ == '__main__':
    G, D = Generator(), Discriminator()
    p2p = Pix2pix()
    G.summary(line_length=120)
    D.summary(line_length=120)

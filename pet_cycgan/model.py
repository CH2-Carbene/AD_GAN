import time,datetime,os,pickle
import numpy as np
import tensorflow as tf
from units.base import show,visualize,calc_metric
from units.prep import normalize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate, ZeroPadding3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from .losses import L1_loss, L2_loss
from .layers import G_conv3d, G_deconv3d, Reslayer
K_INITER = "he_normal"


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
    ks = 4
    depth = 3
    nf_start = 16

    x = inputs
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
    return Model(inputs=inputs, outputs=outputs, name='Discriminator')

# def add(f1,f2):
    # return lambda x:f1(*x)+f2(*x)


def showState(d:dict):
    for x,y in d.items():
        show(f"{x} : {y:6f}")
    show("")

class Cycgan_pet:
    def __init__(self,input_shape=(128,128,128,1),lamda=10,example_data=None):
        """
        Cycgan with paired data. Disc for True and fake image, and L1 loss for cycle consistency.
        """

        self.lamda = lamda
        self.input_shape=input_shape
        self.calc_cyc_loss = lambda inp, cyc: L1_loss(inp, cyc)
        self.calc_gan_dis_loss = lambda dis_out: L1_loss(dis_out, 1)
        self.calc_gan_loss = lambda cyc_loss, dis_out: self.lamda * \
            cyc_loss+self.calc_gan_dis_loss(dis_out)

        self.calc_dis_loss = lambda fak, tru: (
            L2_loss(fak, 0) + L2_loss(tru, 1)) / 2

        # curr_lr = 0.0002*(200-max(epoch, 100))/100
        
        self.G1, self.G2 = Generator(input_shape), Generator(input_shape)
        self.DA, self.DB = Discriminator(input_shape), Discriminator(input_shape)
        
        self.G1_op = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.G2_op = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.DA_op = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.DB_op = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.outputs=["G1_loss","G2_loss","DA_loss","DB_loss","cyc_loss_A","cyc_loss_B","tot_cyc_loss"]
        
        self.applyop=lambda tape,op,model,loss:op.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables))

        self.log_dir="logs/" + f"lamda{self.lamda}_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path=f"{self.log_dir}/Pet_cyc"
        self.prev_loss=np.inf

        if example_data is not None:
            x,y=example_data[0],example_data[1]
            x,y=tf.reshape(x,input_shape[:-1]),tf.reshape(y,input_shape[:-1])
            self.example=[x,y]
        else:self.example=None

    def load_model(self,model_path):
        self.G1.load_weights(model_path+"/G1.h5")
        self.G2.load_weights(model_path+"/G2.h5")
        self.DA.load_weights(model_path+"/DA.h5")
        self.DB.load_weights(model_path+"/DB.h5")

    def generate_images(self, generate_img:tf.Tensor=None, save_path=None):
        if generate_img is None:
            generate_img=self.example
        imgA,imgB=generate_img
        G1,G2=self.G1,self.G2

        fakeB = tf.reshape(G1(imgA[tf.newaxis,...,tf.newaxis], training=False), imgB.shape)
        cycA = tf.reshape(G2(fakeB[tf.newaxis,...,tf.newaxis], training=False), imgA.shape)
        fakeA = tf.reshape(G2(imgB[tf.newaxis,...,tf.newaxis], training=False), imgB.shape)
        cycB = tf.reshape(G1(fakeA[tf.newaxis,...,tf.newaxis], training=False), imgA.shape)

        display_list = [imgA,imgB,fakeA,fakeB,cycA,cycB]
        # title = ['Input Image', 'Ground Truth', 'Predicted Image']
        visualize(display_list, save_path=save_path)

    def save_checkpoint(self,step,now_loss):
        G1, G2, DA, DB = self.G1, self.G2, self.DA, self.DB
        path,prev_loss=self.path,self.prev_loss
        save_path = f"{path}/step_{step+1:03d}"
        
        os.makedirs(save_path, exist_ok=True)

        if self.example is not None:
            self.generate_images(save_path=f"{save_path}/show.png")
        G1.save(f"{save_path}/G1.h5")
        G2.save(f"{save_path}/G2.h5")
        DA.save(f"{save_path}/DA.h5")
        DB.save(f"{save_path}/DB.h5")

        if now_loss < prev_loss:
            G1.save(f"{path}/G1.h5")
            G2.save(f"{path}/G2.h5")
            DA.save(f"{path}/DA.h5")
            DB.save(f"{path}/DB.h5")
            show(f"Validation loss decresaed from {prev_loss:.4f} to {now_loss:.4f}. Models' weights are now saved.")
            self.prev_loss = now_loss
        else:
            show(f"Validation loss did not decrese from {prev_loss:.4f} to {now_loss:.4f}.")

    def eval_result_norm(self,test_ds):
        pA={"MSE":0,"SSIM":0,"PSNR":0,"NMI":0}
        pB={"MSE":0,"SSIM":0,"PSNR":0,"NMI":0}
        num=0
        G1,G2=self.G1,self.G2
        for val_step, (imgA, imgB) in test_ds.enumerate():
            fakeB = G1(imgA, training=False)
            fakeA = G2(imgB, training=False)
            dA=calc_metric(normalize(imgA[0,...,0].numpy()),normalize(fakeA[0,...,0].numpy()))
            dB=calc_metric(normalize(imgB[0,...,0].numpy()),normalize(fakeB[0,...,0].numpy()))
            for x,y in dA.items():
                pA[x]+=y
            for x,y in dB.items():
                pB[x]+=y
            num+=1
        resA,resB={x:y/num for x,y in pA.items()},{x:y/num for x,y in pB.items()}
        return resA,resB

    def eval_result(self,test_ds):
        pA={"MSE":0,"SSIM":0,"PSNR":0,"NMI":0}
        pB={"MSE":0,"SSIM":0,"PSNR":0,"NMI":0}
        num=0
        G1,G2=self.G1,self.G2
        for val_step, (imgA, imgB) in test_ds.enumerate():
            fakeB = G1(imgA, training=False)
            fakeA = G2(imgB, training=False)
            dA=calc_metric(imgA[0,...,0].numpy(),fakeA[0,...,0].numpy())
            dB=calc_metric(imgB[0,...,0].numpy(),fakeB[0,...,0].numpy())
            for x,y in dA.items():
                pA[x]+=y
            for x,y in dB.items():
                pB[x]+=y
            num+=1
        resA,resB={x:y/num for x,y in pA.items()},{x:y/num for x,y in pB.items()}
        return resA,resB

    def test(self, test_ds):
        test_losses = [tf.keras.metrics.Mean() for i in range(7)]
        for val_step, (imgA, imgB) in test_ds.enumerate():
            test_step_loss = self.test_step(imgA, imgB)
            for meti, li in zip(test_losses, test_step_loss):
                meti.update_state(li)
        return [x.result() for x in test_losses]

    def test_step(self,imgA, imgB):
        G1, G2, DA, DB = self.G1, self.G2, self.DA, self.DB

        calc_gan_loss = self.calc_gan_loss
        calc_dis_loss = self.calc_dis_loss
        calc_cyc_loss = self.calc_cyc_loss

        fakeB = G1(imgA, training=False)
        cycleA = G2(fakeB, training=False)

        fakeA = G2(imgB, training=False)
        cycleB = G1(fakeA, training=False)

        imgA_dis = DA(imgA, training=False)
        imgB_dis = DB(imgB, training=False)
        fakeB_dis = DB(fakeB, training=False)
        fakeA_dis = DA(fakeA, training=False)

        cyc_loss_A=calc_cyc_loss(imgA, cycleA)
        cyc_loss_B=calc_cyc_loss(imgB, cycleB)
        tot_cyc_loss = cyc_loss_A+cyc_loss_B

        DA_loss = calc_dis_loss(fakeA_dis, imgA_dis)
        DB_loss = calc_dis_loss(fakeB_dis, imgB_dis)
        
        G1_loss = calc_gan_loss(tot_cyc_loss, fakeB_dis)
        G2_loss = calc_gan_loss(tot_cyc_loss, fakeA_dis)

        return G1_loss,G2_loss,DA_loss,DB_loss,cyc_loss_A,cyc_loss_B,tot_cyc_loss

    def train_step(self, imgA, imgB):
        """
        G1:A->B
        G2:B->A
        """
        G1, G2, DA, DB = self.G1, self.G2, self.DA, self.DB
        G1_op,G2_op,DA_op,DB_op=self.G1_op,self.G2_op,self.DA_op,self.DB_op
        calc_gan_loss = self.calc_gan_loss
        calc_dis_loss = self.calc_dis_loss
        calc_cyc_loss = self.calc_cyc_loss

        with tf.GradientTape(persistent=True) as tape:
            fakeB = G1(imgA, training=True)
            cycleA = G2(fakeB, training=True)

            fakeA = G2(imgB, training=True)
            cycleB = G1(fakeA, training=True)

            imgA_dis = DA(imgA, training=True)
            imgB_dis = DB(imgB, training=True)
            fakeB_dis = DB(fakeB, training=True)
            fakeA_dis = DA(fakeA, training=True)

            cyc_loss_A=calc_cyc_loss(imgA, cycleA)
            cyc_loss_B=calc_cyc_loss(imgB, cycleB)
            tot_cyc_loss = cyc_loss_A+cyc_loss_B

            DA_loss = calc_dis_loss(fakeA_dis, imgA_dis)
            DB_loss = calc_dis_loss(fakeB_dis, imgB_dis)
            
            G1_loss = calc_gan_loss(tot_cyc_loss, fakeB_dis)
            G2_loss = calc_gan_loss(tot_cyc_loss, fakeA_dis)
            
        G1_grad = tape.gradient(G1_loss, G1.trainable_variables)
        G2_grad = tape.gradient(G2_loss, G2.trainable_variables)
        DA_grad = tape.gradient(DA_loss, DA.trainable_variables)
        DB_grad = tape.gradient(DB_loss, DB.trainable_variables)

        G1_op.apply_gradients(zip(G1_grad,G1.trainable_variables))
        G2_op.apply_gradients(zip(G2_grad,G2.trainable_variables))
        DA_op.apply_gradients(zip(DA_grad,DA.trainable_variables))
        DB_op.apply_gradients(zip(DB_grad,DB.trainable_variables))

        return G1_loss,G2_loss,DA_loss,DB_loss,cyc_loss_A,cyc_loss_B,tot_cyc_loss

    def train(self, train_ds, val_ds, batch_size=1, epoches:int=None, steps: int = None, val_time=100):

        history = {'train':[],'valid':[]}
        if steps is None:
            steps = len(train_ds)*400//batch_size
        val_freq = steps//val_time
        train_losses = [tf.keras.metrics.Mean() for i in range(7)]
        
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

            G1_loss,G2_loss,DA_loss,DB_loss,cyc_loss_A,cyc_loss_B,tot_cyc_loss = train_step_loss

            showState({"G1_loss":G1_loss,"G2_loss":G2_loss,
            "DA_loss":DA_loss,"DB_loss":DB_loss,"cyc_loss_A":cyc_loss_A,"cyc_loss_B":cyc_loss_B,"tot_cyc_loss":tot_cyc_loss})

            show(f'Time taken for 1 steps: {time.time()-start:.2f} sec\n')

            if (step+1) % val_freq == 0 or step == 0:

                show(f"Val_step: {(step+1)//val_freq}/{val_time}")

                show("Train loss:")
                this_train_losses= [x.result() for x in train_losses]
                G1_loss,G2_loss,DA_loss,DB_loss,cyc_loss_A,cyc_loss_B,tot_cyc_loss= this_train_losses
                showState({"G1_loss":G1_loss,"G2_loss":G2_loss,"DA_loss":DA_loss,"DB_loss":DB_loss,"cyc_loss_A":cyc_loss_A,"cyc_loss_B":cyc_loss_B,"tot_cyc_loss":tot_cyc_loss})

                show("Val loss:")
                this_val_losses=self.test(val_ds)
                G1_loss,G2_loss,DA_loss,DB_loss,cyc_loss_A,cyc_loss_B,tot_cyc_loss = this_val_losses
                showState({"G1_loss":G1_loss,"G2_loss":G2_loss,"DA_loss":DA_loss,"DB_loss":DB_loss,"cyc_loss_A":cyc_loss_A,"cyc_loss_B":cyc_loss_B,"tot_cyc_loss":tot_cyc_loss})

                self.save_checkpoint(step,G1_loss+G2_loss)

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
    cyc=Cycgan_pet()
    G.summary(line_length=120)
    D.summary(line_length=120)
    tf.keras.utils.plot_model(G,to_file="G.png",show_shapes=True)
    tf.keras.utils.plot_model(D,to_file="D.png",show_shapes=True)
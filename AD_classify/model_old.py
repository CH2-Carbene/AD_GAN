import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers,optimizers,losses,Sequential,metrics
from tensorflow.keras.layers import Input, Flatten, Conv3D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate, ZeroPadding3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

K_INITER = "he_normal"

# def Classify():
#     '''
#     Classify model between NC/MCI
#     '''
    
#     mod=2
#     Nfilter_start = 16
#     depth = 3
#     ks = 4
#     embedding_channel=4
#     inputs = [Input((128,128,128,1), name=f'input_image_{i}') for i in range(mod)]
#     # inputs = [Input((128,128,128,1), name=f'input_image_0')]

#     def encoder_step(layer, Nf, norm=True):
#         x = Conv3D(Nf, kernel_size=ks, strides=2, kernel_initializer=K_INITER, padding='same')(layer)
#         if norm:
#             x = InstanceNormalization()(x)
#         x = LeakyReLU()(x)
#         x = Dropout(0.2)(x)
        
#         return x

#     x_list = [x for x in inputs]
#     for i in range(len(x_list)):
#         x=x_list[i]
#         for d in range(depth):
#             if d==0:
#                 x = encoder_step(x, Nfilter_start*np.power(2,d), False)
#             else:
#                 x = encoder_step(x, Nfilter_start*np.power(2,d))

#         x = ZeroPadding3D()(x)
#         x = Conv3D(Nfilter_start*(2**depth), ks, strides=1, padding='valid', kernel_initializer=K_INITER)(x) 
#         x = InstanceNormalization()(x)
#         x = LeakyReLU()(x)
#         # x = Conv3D(embedding_channel, ks, strides=1, padding='valid', kernel_initializer=K_INITER)(x) 
#         # x = LeakyReLU()(x)
        
#         x_list[i]=x

#     x = Concatenate()(x_list)
    
#     x = ZeroPadding3D()(x)
#     last = Conv3D(1, ks, strides=1, padding='valid', kernel_initializer=K_INITER, name='output_classify')(x)

#     return Model(inputs=inputs, outputs=last, name='Classify')

#conv3d
def Conv3D_U(channel):
    return layers.Conv3D(channel,3,padding='same')

#BN+activate(relu)
def BN_AC():
    return Sequential([
        InstanceNormalization(),
        layers.Activation("relu"),
    ])

#conv3d+layerNormalization
def Conv3D_BN(channel,dp_rate=0):
    return Sequential([
        Conv3D_U(channel),
        BN_AC(),
        layers.Dropout(dp_rate),
    ])

#Conv3D_Pooling1
def Conv3D_P1(channel):
    return layers.Conv3D(channel,3,strides=(2,2,2))
#Conv3D_Pooling2
def Conv3D_P2(channel):
    return layers.Conv3D(channel,3,strides=(2,2,2),padding='same')
#Conv3D_Pooling+BatchNormalization
# why dropout before BatchNormalization?
def Conv3D_PBN(channel,dp_rate=0):
    return Sequential([
        Conv3D_P2(channel),
        layers.Dropout(dp_rate),
        BN_AC()
    ])

def Merge():
    return layers.Concatenate()
def Liner(units,activation=None):
    return layers.Dense(units,activation=activation)

def CNN3D(cls_num=2,mods=2):
    ''' Return a 3D-CNN model for classification/regression.
    input shape:(42,50,42,1)
    output shape:(cls_num)
    Args:
      cls_num: int, should be >=1. When cls_num is 1, it's a regression model.
    '''
    # is_reg=True if cls_num==1 else False
    def conv_layers(inputs):
            
        L0=Sequential([
            Conv3D_BN(15),
            Conv3D_P1(15)
        ])#,name="Block0")

        L1=Sequential([
            Conv3D_BN(15,0.2),
            Conv3D_U(15)
        ])#,name="Block1")
        M1=Merge()

        L2=Sequential([
            BN_AC(),
            Conv3D_PBN(25,0.2),
            Conv3D_U(25)
        ])#,name="Block2")
        R2=Conv3D_P2(15)
        M2=Merge()

        L3=Sequential([
            BN_AC(),
            Conv3D_PBN(35,0.2),
            Conv3D_U(35)
        ])#,name="Block3")
        R3=Conv3D_P2(25)
        M3=Merge()
        
        L4=Sequential([
            BN_AC(),
            layers.Conv3D(30,3,padding='valid'),
            layers.Conv3D(30,3,padding='valid')
        ])#,name="Block4")

        l0_out=L0(inputs)

        l1_x=L1(l0_out)
        l1_y=l0_out
        l1_out=M1([l1_x,l1_y])

        l2_x=L2(l1_out)
        l2_y=R2(l1_out)
        l2_out=M2([l2_x,l2_y])

        l3_x=L3(l2_out)
        l3_y=R3(l2_out)
        l3_out=M3([l3_x,l3_y])

        l4_x=L4(l3_out)
        outputs=layers.Flatten()(l4_x)
        return outputs

    def line_layers(inputs):

        FC=Sequential([
            Liner(300,'relu'),
            layers.Dropout(0.2),
            Liner(50,'relu'),
            Liner(cls_num)
        ])#,name="FC")
        return FC(inputs)
    CLF=layers.Softmax()

### network constructure
    inputs = Input(shape=(mods,42,50,42,1), dtype='float32')
    
    
    fe_list=[conv_layers(inputs[:,mi,...])for mi in range(mods)]
    lfc=Merge()(fe_list)
    # print(lfc.shape)
    x2=line_layers(lfc)
    sfx=CLF(x2)

    opt=optimizers.Adadelta()
    
    loss_func=losses.SparseCategoricalCrossentropy()
    metric=metrics.SparseCategoricalAccuracy()

    model=Model(inputs=inputs, outputs=sfx)
    model.compile(optimizer=opt,loss=loss_func,metrics=metric)
    return model
import datetime,os
# class Patch_clf:
    
#     def __init__(self,mods=2):
#         self.model=[CNN3D(2,mods)for i in range(27)]
#         self.log_dir = "logs/" + \
#             f"CLF_" + \
#             datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         self.path = f"{self.log_dir}/Pix2pix"

#     def train(self,train_ds,val_ds,epoches=200):
#         ### ds:([T1,T2],label)

#         tdss=[]
#         vdss=[]
#         for i in range(3):
#             for j in range(3):
#                 for k in range(3):
#                     tdss.append(train_ds.map(lambda imgs,label:((tf.slice(img,begin=(i*21,j*25,k*21),size=(42,50,42))for img in imgs),label)))
#                     vdss.append(val_ds.map(lambda imgs,label:((tf.slice(img,begin=(i*21,j*25,k*21),size=(42,50,42))for img in imgs),label)))
#         # train_ds=[[[for i in range(3)]for j in range(3)]for k in range(3)]
#         print("start training...")
#         for e in range(epoches):
#             for i in range(27):
#                 model=self.model[i]
#                 model.fit(tdss[i],batch_size=4,validation_data=vdss[i])
#                 if e%4==0:
#                     self.save_checkpoint(e)
#         # y_predit=[self.model[i].predict(vdss[i])for i in range(27)]
#         self.test(val_ds)
#         # print("Val accuracy={}".format(self.test(val_ds)))

#     def save_checkpoint(self, step):
#         path = self.path
#         save_path = f"{path}/step_{step+1:03d}"

#         os.makedirs(save_path, exist_ok=True)

#         for i,mi in enumerate(self.model):
#             mi.save(f"{save_path}/c_{i}.h5")

#     def test(self,test_ds):
#         tdss=[]
#         label=test_ds.map(lambda ings,label:label)
#         for i in range(3):
#             for j in range(3):
#                 for k in range(3):
#                     tdss.append(test_ds.map(lambda imgs,label:((tf.slice(img,begin=(i*21,j*25,k*21),size=(42,50,42))for img in imgs),label)))
        
#         n=len(label)
#         vote=np.zeros((n,2),dtype=np.int32)
#         for i in range(27):
#             model=self.model[i]
#             y_predit=model.predict(tdss[i])
#             tot=metrics.sparse_categorical_accuracy(label,y_predit)
            
#             acc=sum(tot)/len(tot)
#             print(f"Test model {i} acc: {acc}")
#             yp_lab=np.array([np.argmax(i)for i in y_predit])
#             for i,li in enumerate(yp_lab):
#                 vote[i][li]+=1

#         vote_result=np.argmax(vote,axis=1)
#         tot_acc=(label==vote_result)
#         acc=sum(tot_acc)/len(tot_acc)
#         print("Test tot acc: ",acc)
#         return vote_result
        # return acc
class CNN_clf:
    
    def __init__(self,mods=2):
        self.model=[CNN3D(2,mods)for i in range(1)]
        self.log_dir = "logs/" + \
            f"CLF_" + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = f"{self.log_dir}/Pix2pix"

    def split_train_ds(ds,batch_size=32):
        '''
        Split train dataset into 42*50*42 patches
        '''
        tds=None
        for i in range(3):
            for j in range(3):
                for k in range(3):

                    ts=ds.map(lambda imgs,label:(imgs[:,i*21:i*21+42,j*25:j*25+50,k*21:k*21+42,tf.newaxis],label))
                    
                    if tds is None:
                        tds=ts
                    else:
                        tds=tds.concatenate(ts)

        tds=tds.shuffle(114514).batch(batch_size)
        return tds
        
    def split_test_ds(ds,batch_size=32):
        '''
        Split test dataset into 42*50*42 patches
        '''
        tds=None
        for i in range(3):
            for j in range(3):
                for k in range(3):

                    ts=ds.map(lambda imgs,label:(imgs[:,i*21:i*21+42,j*25:j*25+50,k*21:k*21+42,tf.newaxis],label))
                    
                    if tds is None:
                        tds=ts
                    else:
                        tds=tds.concatenate(ts)
        # train_ds=[[[for i in range(3)]for j in range(3)]for k in range(3)]

        tds=tds.batch(batch_size)
        return tds

    def save_checkpoint(self, step):
        path = self.path
        save_path = f"{path}/step_{step+1:03d}"

        os.makedirs(save_path, exist_ok=True)

        for i,mi in enumerate(self.model):
            mi.save(f"{save_path}/c_{i}.h5")

    def train(self,train_ds,val_ds,batch_size=32,epoches=200):
        # print(tf.compat.v1.executing_eagerly())

        # for i,j in tdss[0]:
        #     tx.append(i.numpy())
        #     ty.append(j.numpy())
        # tx=np.array(tx)
        # ty=np.array(ty)
        # self.model[0].fit(tx,ty)
        # print("Pass!")
        # print(self.model[0].summary())
        tds,vds=CNN_clf.split_train_ds(train_ds,batch_size=batch_size),CNN_clf.split_test_ds(val_ds,batch_size=batch_size)

        print("start training...")
        for e in range(epoches):
            print(f"Epoch {e}:")
                # print("I: ",i)
                # print(tdss[i])
            model=self.model[0]
            model.fit(tds,validation_data=vds)
            self.save_checkpoint(e)
        # y_predit=[self.model[i].predict(vdss[i])for i in range(27)]
        self.test(val_ds,batch_size=batch_size)
        # print("Val accuracy={}".format(self.test(val_ds)))

    def test(self,test_ds,batch_size=32):
        tdss=[]
        label=list(test_ds.map(lambda imgs,label:label).as_numpy_iterator())
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    tdss.append(test_ds.map(lambda imgs,label:(imgs[:,i*21:i*21+42,j*25:j*25+50,k*21:k*21+42,tf.newaxis],label)).batch(batch_size))
        # tds=tds.batch(batch_size)
        n=len(label)
        vote=np.zeros((n,2),dtype=np.int32)
        for i in range(27):
            model=self.model[0]
            y_predit=model.predict(tdss[i])
            tot=metrics.sparse_categorical_accuracy(label,y_predit)
            
            acc=sum(tot)/len(tot)
            print(f"Test model {i} acc: {acc}")
            yp_lab=np.array([np.argmax(i)for i in y_predit])
            for i,li in enumerate(yp_lab):
                vote[i][li]+=1

        vote_result=np.argmax(vote,axis=1)
        tot_acc=(label==vote_result)
        acc=sum(tot_acc)/len(tot_acc)
        print("Test tot acc: ",acc)
        return vote_result

# # BUFFER_SIZE=20
# N_CPU=20
# def load_np_data(filename,load_mods=["T1","T2"],argu=False):
#     # st=(17, 26, 5)
#     # ed=(-18, -22, -30)
#     if type(filename)!=str:
#         filename=filename.decode()

#     data=np.load(filename,mmap_mode="r")
#     imgs=np.array([data[md.decode()if type(md)!=str else md]for md in load_mods])
#     label=0. if data["label"]=="NC" else 1.
#     # data=data[:,st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
#     return (tf.convert_to_tensor(imgs),tf.convert_to_tensor(label))

# train_load=lambda filename:tf.numpy_function(func=load_np_data,inp=[filename,["T1","T2"]],Tout=(tf.float32,tf.float32))
# test_load=lambda filename:tf.numpy_function(func=load_np_data,inp=[filename,["T1","T2"]],Tout=(tf.float32,tf.float32))

# def get_train_ds(train):
#     # train_dataset=[]
#     # for t in tqdm(train):
#         # train_dataset.append(load_image_train(t))
#     # train_dataset=np.array(train_dataset)
#     # train_dataset = list(map(load_image_train,train))
    
#     train_dataset = tf.data.Dataset.from_tensor_slices(train)
#     # print(train_dataset)
#     # train_dataset=load_image_train(train)
#     train_dataset = train_dataset.map(map_func=train_load,num_parallel_calls=N_CPU)
#     # train_dataset = train_dataset.shuffle(BUFFER_SIZE,seed=114514)
#     return train_dataset
# def get_test_ds(test):
#     # train_dataset=[]
#     # for t in tqdm(train):
#         # train_dataset.append(load_image_train(t))
#     # train_dataset=np.array(train_dataset)
#     # train_dataset = list(map(load_image_train,train))
    
#     test_dataset = tf.data.Dataset.from_tensor_slices(test)
#     # print(train_dataset)
#     # train_dataset=load_image_train(train)
#     test_dataset = test_dataset.map(map_func=test_load,num_parallel_calls=N_CPU)
#     # train_dataset = train_dataset.shuffle(BUFFER_SIZE,seed=114514)
#     return test_dataset

# from sklearn.model_selection import train_test_split
# import pandas as pd


# def prep_data(data,plist=None):
#     try:
#         ds=load_subject(f"{DATAPATH}/{data}",T1_name="T1.nii.gz",others_name=["T2.nii.gz"])
#         if plist is not None:
#             for pid,label in plist:
#                 if data.find(pid)!=-1:
#                     ds["label"]=label
#         np.savez(f"{NEWPATH}/{data}",**ds)
#         print(f"{data} finish!")
#     except Exception as e:
#         raise Exception(f"{data} Failed: {e}\n")

# def make_patch(src,dst,info):
#     data=[f"{src}/{img_dir}/T1.nii.gz" for img_dir in os.listdir(DATA_ORI)]

#     df=pd.read_csv(CSV_PATH,dtype=str,keep_default_na=False)
#     plist=[(value["PID"],value["diagonsis"]) for value in df[df["diagonsis"]!=""][["PID","diagonsis"]].iloc()]
#     for fname in data:


# if __name__ == '__main__':
#     DATA_ORI="/public_bme/data/gujch/ZS_t1_full/05_ZS/result"
#     PATCH_ORI="/public_bme/data/gujch/ZS_t1_full/patches"
#     CSV_PATH="/public_bme/data/gujch/ZS_t1_full/Diagnosis Information.csv"
    
#     df=pd.read_csv(CSV_PATH,dtype=str,keep_default_na=False)
#     plist=[(value["PID"],value["diagonsis"]) for value in df[df["diagonsis"]!=""][["PID","diagonsis"]].iloc()]
#     make_patch(DATA_ORI,PATCH_ORI,CSV_PATH)

#     train_val,test=train_test_split(
#         data,test_size=0.1,random_state=1919810
#     )
#     train,val=train_test_split(
#         train_val,test_size=0.1,random_state=114514
#     )
#     print(f"Train len: {len(train)}")
#     print(f"Val len: {len(val)}")
#     print(f"Test len: {len(test)}")
#     train_ds,val_ds,test_ds=get_train_ds(train),get_test_ds(val),get_test_ds(test)
#     # print(np.load(train[0])["label"])
#     # e_img,e_lab=load_np_data(train[0],argu="False")
#     # print(e_img.shape)
#     # for i in range(3):
#     #     for j in range(3):
#     #         for k in range(3):
#     #             print(e_img[:,i*21:i*21+42,j*25:j*25+50,k*21:k*21+42].shape)
    
#     # npds=np.array([e_img[:,i*21:i*21+42,j*25:j*25+50,k*21:k*21+42]])
#     # nplb=np.array([0.],dtype=np.float32)
#     # print("NPDS: ",npds.shape)

    
#     # model=CNN3D()
#     # np.array(list(train_ds.as_numpy_iterator()))
    

#     # print(model.summary())

#     # model.fit(npds,nplb,batch_size=4)

#     # input()
    
#     clf=CNN_clf()
#     # clf.test(train_ds,32)
#     clf.train(train_ds,val_ds,4,200)
#     clf.test(test_ds)
#     # m=CNN3D(2,2)
#     # m.summary(line_length=120)
    
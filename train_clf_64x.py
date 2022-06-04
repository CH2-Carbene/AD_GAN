import numpy as np
import tensorflow as tf
from AD_classify.model import CNN_clf
from units.base import visualize
from units.prep import random_jitter
from sys import argv
# BUFFER_SIZE=20
N_CPU=20
DELTA_LIST=argv[1:]#["delta1","delta2","delta3"]
LOAD_MODS=["T1"]+DELTA_LIST
BUFFER_SIZE=200
def load_np_data(dirname,load_mods=["T1"],argu=False,select_shape=(128,128,128)):
    # st=(17, 26, 5)
    # ed=(-18, -22, -30)
    if type(dirname)!=str:
        dirname=dirname.decode()

    pat,lab=[],[]
    for filename in os.listdir(dirname):
        data=np.load(os.path.join(dirname,filename),mmap_mode="r")
        imgs=np.array([data[md.decode()if type(md)!=str else md]for md in load_mods])
        # data=data[:,st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
        if argu==False:
            imgs=random_jitter(imgs,only_select=True,select_shape=select_shape)
        else:
            imgs=random_jitter(imgs,only_select=False,select_shape=select_shape)
        
        label=0. if data["label"]=="NC" else 1.
        # print("label=",label)
        pat.append(tf.convert_to_tensor(imgs[...,tf.newaxis]))
        lab.append(tf.convert_to_tensor(label))
    # data=data[:,st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
    # print(len(pat))
    return pat,lab


train_load=lambda filename:tf.numpy_function(func=load_np_data,inp=[filename,LOAD_MODS,True,(64,64,64)],Tout=(tf.float32,tf.float32))
test_load=lambda filename:tf.numpy_function(func=load_np_data,inp=[filename,LOAD_MODS,False,(64,64,64)],Tout=(tf.float32,tf.float32))



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
    # train_dataset = train_dataset.unbatch()
    # train_dataset = train_dataset.shuffle()
    # train_dataset = train_dataset.shuffle(BUFFER_SIZE,seed=114514)

    return train_dataset

def get_test_ds(test):
    # train_dataset=[]
    # for t in tqdm(train):
        # train_dataset.append(load_image_train(t))
    # train_dataset=np.array(train_dataset)
    # train_dataset = list(map(load_image_train,train))
    
    test_dataset = tf.data.Dataset.from_tensor_slices(test)
    # print(train_dataset)
    # train_dataset=load_image_train(train)
    test_dataset = test_dataset.map(map_func=test_load,num_parallel_calls=N_CPU)
    # train_dataset = train_dataset.shuffle(BUFFER_SIZE,seed=114514)
    return test_dataset

from sklearn.model_selection import train_test_split
import os
import pandas as pd
from units.dataloader import load_subject
from units.get_delta import get_half_delta
from pet_cycgan.model import Cycgan_pet

def prep_data(data,plist=None):
    try:
        ds=load_subject(f"{DATAPATH}/{data}",T1_name="T1.nii.gz",others_name=["T2.nii.gz"])
        if plist is not None:
            for pid,label in plist:
                if data.find(pid)!=-1:
                    ds["label"]=label
        np.savez(f"{NEWPATH}/{data}",**ds)
        print(f"{data} finish!")
    except Exception as e:
        raise Exception(f"{data} Failed: {e}\n")


ROOT="/public_bme/data/gujch/ZS_t1_full/"
# ROOT="datasets/ZS_t1_full/"

DATA_ORI=ROOT+"05_ZS/result/"
PATCH_ORI=ROOT+"patches_full/"
CSV_PATH=ROOT+"Diagnosis Information.csv"
PATCH_SIZE=(128,128,128)
PATCH_NUM=(3,3,3)
MODEL_PATH="/hpc/data/home/bme/v-gujch/work/AD_GAN/logs/T1-FA_lamda10.0_AdamOpt_CH_Res_20220517-021959/Pet_cyc/step_133400/"
# MODEL_PATH="logs/T1_FA_l=10/"

# MODEL_PATH="pet_cycgan"

def save_patch(ds,dst):
    img,label=ds["T1"],ds["label"]
    step_size=(np.array(img.shape)-np.array(PATCH_SIZE))//(np.array(PATCH_NUM)-1)
    cyc=Cycgan_pet(G_net="CH_Res")
    cyc.load_model(MODEL_PATH)
    os.makedirs(dst,exist_ok=True)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                pat_id=i*9+j*3+k
                # print(f"{dst}/patch_{pat_id}")
                if os.path.exists(f"{dst}/patch_{pat_id}.npz"):
                    print(f"{dst}/patch_{pat_id}.npz exist, continue")
                    continue
                
                st=step_size*np.array((i,j,k))
                ed=st+np.array(PATCH_SIZE)
                pat=img[st[0]:ed[0],st[1]:ed[1],st[2]:ed[2]]
                _,_,delta=get_half_delta(cyc.G1,cyc.G2,pat[np.newaxis,...,np.newaxis])
                # print(delta[0].shape)
                pds={f"delta{h+1}":delta[h] for h in range(3)}
                pds["T1"]=pat
                pds["label"]=label
                
                np.savez(f"{dst}/patch_{pat_id}",**pds)
                
    

def make_patch(src,dst,info):
    data=[f"{src}/{img_dir}" for img_dir in os.listdir(DATA_ORI)]

    df=pd.read_csv(CSV_PATH,dtype=str,keep_default_na=False)
    plist=[(value["PID"],value["diagonsis"]) for value in df[df["diagonsis"]!=""][["PID","diagonsis"]].iloc()]

    datalist=[]
    for fname in data:
        ds=load_subject(fname,others_name=[])
        for pid,label in plist:
            if fname.find(pid)!=-1:

                ds["label"]=label
                save_patch(ds,dst+pid)
                datalist.append(dst+pid)
                break
    return datalist

def get_fb(ds):
    t,f=0,0
    for img,lab in ds.as_numpy_iterator():
        lab=lab[0]
        # print(lab)
        t+=lab
        f+=1-lab
    return t,f

if __name__ == '__main__':

    print("Mods:","_".join(LOAD_MODS))
    df=pd.read_csv(CSV_PATH,dtype=str,keep_default_na=False)
    plist=[(value["PID"],value["diagonsis"]) for value in df[df["diagonsis"]!=""][["PID","diagonsis"]].iloc()]
    data=make_patch(DATA_ORI,PATCH_ORI,CSV_PATH)
    print(data)
    
    test_rate=0.2
    train,test=train_test_split(
        data,test_size=test_rate,random_state=1919810
    )
    
    print(f"Train len: {len(train)}")
    print(f"Test len: {len(test)}")
    train_ds,test_ds=get_train_ds(train),get_test_ds(test)
    
    for ds,name in zip((train_ds,test_ds),("Train","Test")):
        t,f=get_fb(ds)
        print(name+":","MCI="+str(t),"NC="+str(f))
    
    clf=CNN_clf(LOAD_MODS,model="CNN3D_64x")
    # clf.test(train_ds,32)
    clf.train(train_ds,test_ds,32,200)
    # clf.test(test_ds,32)
    # m=CNN3D(2,2)
    # m.summary(line_length=120)
    
# %%
import tensorflow as tf
import numpy as np
import nibabel as nib
# from pet_cycgan.model import G
from units.base import vis_img,vis_img_delta
from tensorflow.keras.models import Model
from units.prep import normalize

# from pet_cycgan.model import Cycgan_pet

def get_delta(G1,G2,imgA,imgB,tms=3):
    SA_A,SA_B=[imgA],[]
    SB_A,SB_B=[],[imgB]
    # p2pgan.generate_images([tip[...,0],fip[...,0]])
    for i in range(tms):
        SA_B.append(G1(SA_A[-1]))
        SA_A.append(G2(SA_B[-1]))
        SB_A.append(G2(SB_B[-1]))
        SB_B.append(G1(SB_A[-1]))
    d_A=[fakei-imgA for fakei in SA_A[1:]]
    D_B=[fakei-imgB for fakei in SA_B[:]]
    D_A=[fakei-imgA for fakei in SB_A[:]]
    d_B=[fakei-imgB for fakei in SB_B[1:]]
    return SA_A[1:],SB_A,d_A,D_A,SA_B,SB_B[1:],d_B,D_B

def get_half_delta(G1,G2,imgA,tms=3):
    SA_A,SA_B=[imgA],[]
    # SB_A,SB_B=[],[imgB]
    # p2pgan.generate_images([tip[...,0],fip[...,0]])
    for i in range(tms):
        SA_B.append(G1(SA_A[-1]))
        SA_A.append(G2(SA_B[-1]))
        # SB_A.append(G2(SB_B[-1]))
        # SB_B.append(G1(SB_A[-1]))
    SA_A=SA_A[1:]

    img_mask=(imgA[0,...,0]==0)
    for i in range(tms):
        SA_A[i]=np.array(SA_A[i][0,...,0])
        SA_A[i][img_mask]=0
    for i in range(tms):
        SA_B[i]=np.array(SA_B[i][0,...,0])
        SA_B[i][img_mask]=0

    d_A=[fakei-imgA[0,...,0] for fakei in SA_A]
    # D_B=[fakei-imgB for fakei in SA_B[:]]
    # D_A=[fakei-imgA for fakei in SB_A[:]]
    # d_B=[fakei-imgB for fakei in SB_B[1:]]
    return SA_A,SA_B,d_A
    # return SA_A[1:],SB_A,d_A,D_A,SA_B,SB_B[1:],d_B,D_B

def half_delta_full(G1,G2:Model,img:np.ndarray,shape=(128,128,128)):
    # shape=()
    p,q,r=img.shape
    overlap=np.array(shape)//2
    sum,tms=np.zeros_like(img),np.zeros_like(img)

    SA_A,SA_B,d_A=[np.zeros((3,)+img.shape)for i in range(3)]
    
    i,fi=0,0
    while fi==0:
        
        if i+shape[0]>=p:
            i=p-shape[0]
            fi=1
        # print("i: ",i)
        j,fj=0,0
        while fj==0:
            
            if j+shape[1]>=q:
                j=q-shape[1]
                fj=1

            k,fk=0,0
            while fk==0:
                
                if k+shape[2]>=r:
                    k=r-shape[2]
                    fk=1

                tSA_A,tSA_B,td_A=get_half_delta(G1,G2,img[np.newaxis,i:i+shape[0],j:j+shape[1],k:k+shape[2],np.newaxis])

                tSA_A,tSA_B,td_A=list(map(np.array,[tSA_A,tSA_B,td_A]))

                SA_A[:,i:i+shape[0],j:j+shape[1],k:k+shape[2]]+=tSA_A
                SA_B[:,i:i+shape[0],j:j+shape[1],k:k+shape[2]]+=tSA_B
                d_A[:,i:i+shape[0],j:j+shape[1],k:k+shape[2]]+=td_A
                # img[np.newaxis,i:i+shape[0],j:j+shape[1],k:k+shape[2],np.newaxis][0,...,0]

                tms[i:i+shape[0],j:j+shape[1],k:k+shape[2]]+=1

                k+=overlap[2]
            j+=overlap[1]
        i+=overlap[0]

    for Si in [SA_A,SA_B,d_A]:
        for i in range(3):
            Si[i]/=tms
    
    return SA_A,SA_B,d_A


def generate_delta(model,in_path,out_path):
    G1,G2=model.G1,model.G2
    nimg=nib.load(f"{in_path}")
    img=normalize(nimg.get_fdata())
    afi,hed=nimg.affine.copy(),nimg.header.copy()

    # img=nimg.get_fdata()#[17:-18,26:-22,5:-30]
    t1=img
    SA_A,SA_B,d_A=half_delta_full(G1,G2,t1,model.input_shape[:-1])

    # vis_img(list(SA_A))
    # vis_img(list(SA_B))
    # vis_img_delta(list(d_A))

    ori_nii=nib.Nifti1Image(img,afi,hed)
    nib.save(ori_nii,rf"{out_path}/T1_ori.nii.gz")

    for i in range(1,4):
        SA_A_nii = nib.Nifti1Image(SA_A[i-1],afi,hed)
        nib.save(SA_A_nii,rf"{out_path}/Fake_T1_{i}.nii.gz")

    for i in range(1,4):
        SA_B_nii = nib.Nifti1Image(SA_B[i-1],afi,hed)
        nib.save(SA_B_nii,rf"{out_path}/Fake_FA_{i}.nii.gz")

    for i in range(1,4):
        delta_nii = nib.Nifti1Image(d_A[i-1],afi,hed)
        nib.save(delta_nii,rf"{out_path}/delta_{i}.nii.gz")
    # for i in enumerate(d_A):
    # nib.save(out_path)
    return SA_A,SA_B,d_A

#%%

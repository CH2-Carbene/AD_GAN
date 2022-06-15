import os
# import tensorflow as tf
import pandas as pd
import numpy as np
import multiprocessing
from units.dataloader import load_subject,load_subject_roi
# ,load_image_train,load_image_test

DATAPATH = "datasets/Zhongshan_prep/paired_t2"
NEWPATH="datasets/Zhongshan_prep/npdata_clf"
# if DEBUG:data=data[:10]
# show(data)
# def load_prep_image(image_dir,direct="T1_to_FA"):
#     input_img_name,real_img_name="T1.nii.gz","FA.nii.gz"
#     if direct!="T1_to_FA":
#         input_img_name,real_img_name="FA.nii.gz","T1.nii.gz"
#     input_image, real_image = load_pair(image_dir,input_img_name,real_img_name)
#     # input_image, real_image = np.tanh(input_image), np.tanh(real_image)
#     # input_image, real_image = random_jitter(input_image,real_image)
#     return input_image.astype("float32"), real_image.astype("float32")

# def load_prep_image(image_dir,direct="T1_to_FA"):
#     input_img_name,real_img_name="T1.nii.gz","FA.nii.gz"
#     if direct!="T1_to_FA":
#         input_img_name,real_img_name="FA.nii.gz","T1.nii.gz"
#     input_image, real_image = load_pair(image_dir,input_img_name,real_img_name)
#     # input_image, real_image = np.tanh(input_image), np.tanh(real_image)
#     # input_image, real_image = random_jitter(input_image,real_image)
#     return input_image.astype("float32"), real_image.astype("float32")

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

if __name__ == '__main__':
    data=[f"{imgdir}"for imgdir in os.listdir(DATAPATH)]
    os.makedirs(NEWPATH,exist_ok=True)
    CSV_PATH="datasets/Zhongshan/Diagnosis Information.csv"

    df=pd.read_csv(CSV_PATH,dtype=str,keep_default_na=False)
    plist=[(value["PID"],value["diagonsis"]) for value in df[df["diagonsis"]!=""][["PID","diagonsis"]].iloc()]
    

    pn=40
    mulpool_flt = multiprocessing.Pool(processes=pn)
    fail_set=[]
    for fname in data:
        # prep_data(fname)
        mulpool_flt.apply_async(prep_data,args=(fname,plist),error_callback=lambda e:fail_set.append(e))
    mulpool_flt.close()
    mulpool_flt.join()

    print(f"fail_set: {fail_set}")

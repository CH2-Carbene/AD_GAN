#%%
# %cd pre_procession
# %cd ..

import pandas as pd
import os
ROOT="/public_bme/data/gujch/"
# ROOT="/mnt/c/Users/CH2/Documents/datasets/Zhongshan_prep/"
tsv_PATH=ROOT+"ZS_t1_full/participants.tsv"
DATA_PATH=ROOT+"ZS_t1_full/data/"
OUTPUT_PATH=ROOT+"ZS_t1_full/05_ZS/"
# %%
data=pd.read_csv(tsv_PATH,sep="\t")
for pid in os.listdir(DATA_PATH):
    PID=data[data["participant_id"]==pid[:-4]]["PID"].item()
    os.makedirs(f"{OUTPUT_PATH}/{PID}",exist_ok=True)
    os.system(f"cp {DATA_PATH}/{pid} {OUTPUT_PATH}/{PID}/t1_0.8mm_hs.mgz")
    print(PID)
# %%
# data[data["participant_id"]==pid[:-4]]
# %%
'''debug for wrong mgz
for dir in /public_bme/data/gujch/ZS_t1_full/05_ZS/*
do
  echo $dir
  if [ -d "$dir" ]; then
    cd $dir
    if [ -f "t1_0.8mm_hs.nii.gz" ]; then
      mv t1_0.8mm_hs.nii.gz t1_0.8mm_hs.mgz
    fi
  fi
done
'''
# %%
import subprocess
import tensorflow as tf
import os,sys
from pet_cycgan import Cycgan_pet
from units.get_delta import generate_delta

def show(s,output=None):
    print(s,flush=True)
    print(s,file=sys.stderr,flush=True)
    if output is not None:
        output.append(s+'\n')

def run_sh(cmd,name="unknown",base_dir="tmp",pname="unknown",outputs=[]):
    # tcmd=
    outputs.append(cmd)
    ret=subprocess.run(f"cd {base_dir} && {cmd} 2>&1",shell=True,stdout=subprocess.PIPE,encoding="utf")
    if ret.stdout is not None:
        outputs.append(ret.stdout+'\n')
    if ret.returncode!=0:
        show(f"{pname}.{name}: {cmd.split()[0]} Failed!",outputs)
        raise Exception(f"{name} Error!")
    show(f"{pname}.{name}: {cmd.split()[0]} Successfully!",outputs)

print("import Finish!")
# %%
DATA_PATH="/public_bme/data/gujch/Zhongshan_prep/paired_t2/"

OUT_PATH="/public_bme/data/gujch/Zhongshan_prep/delta/"
SAVE_PATH="/hpc/data/home/bme/v-gujch/work/AD_GAN/logs/lamda1020220416-014127/Pet_cyc/step_54612/"

OUT_PATH="/public_bme/data/gujch/Zhongshan_prep/delta_from_T2/"
SAVE_PATH="/hpc/data/home/bme/v-gujch/work/AD_GAN/logs/lamda1020220429-133914/Pet_cyc/step_88010/"

# SAVE_PATH="logs/54612"
cycmod=Cycgan_pet(input_shape=(128,128,128,1),lamda=10)
cycmod.load_model(SAVE_PATH)
print(cycmod.G1.summary())
#%%
for pid in os.listdir(DATA_PATH):
    print("processing ",pid,"...")
    f_path=DATA_PATH+pid+"/T1.nii.gz"
    fout_path=OUT_PATH+pid
    os.makedirs(fout_path,exist_ok=True)
    generate_delta(cycmod,f_path,fout_path)

    print(pid," finish!")


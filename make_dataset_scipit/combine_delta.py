# %%
import subprocess
import tensorflow as tf
import os,sys
from pet_cycgan.model import Cycgan_pet
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

BASE="/public_bme/data/gujch/Zhongshan_prep/"
TRANS_PATH=BASE+"avg/"
DELTA_PATH=BASE+"delta/"

for pid in os.listdir(TRANS_PATH):
    print("processing ",pid,"...")
    trans_path=TRANS_PATH+pid
    delta_path=DELTA_PATH+pid
    
    sh=lambda cmd,name="unknown",base_dir=trans_path:run_sh(cmd,name=name,base_dir=base_dir,pname=pid)
    sh(f"cp -t {trans_path} {delta_path}/*")
    sh(f"antsApplyTransforms -d 3 -i delta_1.nii.gz -r fnlt.nii.gzWarped.nii.gz -t fnlt.nii.gzInverseWarped.nii.gz -o delta_1_acpc.nii.gz")
    sh(f"antsApplyTransforms -d 3 -i delta_2.nii.gz -r fnlt.nii.gzWarped.nii.gz -t fnlt.nii.gzInverseWarped.nii.gz -o delta_2_acpc.nii.gz")
    sh(f"antsApplyTransforms -d 3 -i delta_3.nii.gz -r fnlt.nii.gzWarped.nii.gz -t fnlt.nii.gzInverseWarped.nii.gz -o delta_3_acpc.nii.gz")
    # os.makedirs(fout_path,exist_ok=True)
    # generate_delta(cycmod,f_path,fout_path)

    print(pid," finish!")

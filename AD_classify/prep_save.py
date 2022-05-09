import os
import nibabel as nib
from os import makedirs
import subprocess
def show(s,output=None):
    print(s,flush=True)
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

DIR="datasets/Zhongshan/Output/subjects"
# makedirs(f"{DIR}",exist_ok=True)
for sub in os.listdir(DIR):
    # makedirs(f"{DIR}/{sub}/ses-M00/t1_linear/",exist_ok=True)
    # makedirs(f"{DIR}/{sub}/ses-M00/pet/preprocessing/group-MCIvsNC/",exist_ok=True)
    T1PATH=f"{DIR}/{sub}/ses-M00/t1_linear/{sub}_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"
    PETPATH=f"{DIR}/{sub}/ses-M00/pet/preprocessing/group-MCIvsNC/{sub}_ses-M00_acq-fdg_pet_space-Ixi549Space_suvr-pons_mask-brain_pet.nii.gz"
    MSKPATH=f"{DIR}/{sub}/ses-M00/pet/preprocessing/group-MCIvsNC/{sub}_ses-M00_T1w_space-Ixi549Space_brainmask.nii.gz"
    t1_img=nib.load(T1PATH).get_fdata()
    pet_img=nib.load(PETPATH).get_fdata()
    # msk_img=nib.load(MSKPATH).get_fdata()
    print(t1_img.shape)
    print(pet_img.shape)
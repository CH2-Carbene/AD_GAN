# %%
import os,glob
import pandas as pd
import nibabel as nib
DATA_PATH="datasets/Zhongshan/pet-mr"
TAR_PATH="datasets/Zhongshan_prep"
CSV_PATH="datasets/Zhongshan/Diagnosis Information.csv"

# %%
df=pd.read_csv(CSV_PATH,dtype=str,keep_default_na=False)
# df.set_index("PID")
pdict={value:key for key,value in df[df["diagonsis"]!=""]["PID"].to_dict().items()}
plist=[(value["PID"],value["diagonsis"]) for value in df[df["diagonsis"]!=""][["PID","diagonsis"]].iloc()]
plist
#%%
df[["diagonsis","PID"]][df["diagonsis"]!=""].to_dict()
# %%
import subprocess

def show(s,output=None):
    print(s,flush=True)
    # print(s,file=sys.stderr,flush=True)
    if output is not None:
        output.append(s+'\n')

def run_sh(cmd,name="unknown",base_dir="tmp",pname="unknown",outputs=[]):
    # tcmd=
    outputs.append(cmd)
    print(cmd)
    ret=subprocess.run(f"cd {base_dir} && {cmd} 2>&1",shell=True,stdout=subprocess.PIPE,encoding="utf")
    if ret.stdout is not None:
        outputs.append(ret.stdout+'\n')
    if ret.returncode!=0:
        show(f"{pname}.{name}: {cmd.split()[0]} Failed!",outputs)
        raise Exception(f"{name} Error!")
    show(f"{pname}.{name}: {cmd.split()[0]} Successfully!",outputs)


# %%
### 1. 拷贝已有图像
### ${TARGET_DIR}/${MOD}/${PID}/img1,img2...
# patient_dict={}
# for dirpath, dirnames, filenames in os.walk("."):
#     pid=os.path.basename(dirpath).split("_")
#     for poss_h in pid:
#         if poss_h in pdict:
#             df_id=pdict[poss_h]
#             diag=df.iloc()[df_id]["diagonsis"]
#             if diag not in patient_dict:patient_dict[diag]=[]
#             patient_dict[diag].append([poss_h,dirpath])

import glob
import shutil
def is_imgdir(dirpath):
    for mName in os.listdir(dirpath):
        if mName.startswith("t1_gre_fsp"):
            return True
    return False

def cpimg(dirpath:str,need_mod:list):
    try:
        mpid=""
        for pid in plist:
            if dirpath.find(pid)!=-1:
                mpid=pid
        if mpid=="":
            raise(Exception(f"pid not found in {dirpath}!"))

        for md,gr in need_mod:
            ptg=os.path.join(TAR_PATH,md,"05ZS",mpid)
            os.makedirs(ptg,exist_ok=True)

            for img in glob.glob(os.path.join(dirpath,gr)):
                # print(ptg,img)
                # if img.find("NIF")!=-1:
                run_sh(f"cp -r -t {ptg} {img}",base_dir=".",pname="cp {img}")
                # if img.endswith("zip"):
                    # run_sh(f"unzip {ptg}/{img}",base_dir=ptg,pname="cp {img}")
                # print(ptg,img)
        return True
    except Exception as e:
        print(e)
        return False

def get_img(need_mod:list):
    succ_cnt,fail_cnt=0,0
    fail_list=[]
    for dirpath, dirnames, filenames in os.walk(DATA_PATH):
        if is_imgdir(dirpath):
            if cpimg(dirpath,need_mod):
                succ_cnt+=1
            else:
                fail_cnt+=1
                fail_list.append(dirpath)
    print("Succ_cnt: ",succ_cnt)
    print("Fail_cnt: ",fail_cnt)
    print("Fail_list: ",fail_list)

need_mod=[("t1","t1_gre_fsp_3d_sag_0.8mm_[0-9]*"),("t2","t2_mx3d_sag_0.8mm_[0-9]*")]
get_img(need_mod)

# %%
### 2. 预处理T1，T2

# %%
### 3. 配准，获得flt数据

# %%
def load_t1_img(srcpath,tgpath):
    os.makedirs(tgpath,exist_ok=True)
    sh=lambda cmd,name="unknown",base_dir=".":run_sh(cmd,name=name,base_dir=base_dir,pname=srcpath,outputs=[])
    srcpath=srcpath+"/Image/t1_gre_fsp_3d_sag_0.8mm_[0-9]*"
    srcpath=sorted(glob.glob(srcpath))[0]
    sh(f"cp -t {tgpath} {srcpath}/*",name="cp",base_dir=srcpath)
    sh(f"dcm2niix -b y -z y -x n -t n -m n -f t1 -o . -s n -v n .",name="dcm2nii",base_dir=tgpath)
    img=nib.load("t1.nii.gz")
    img_arr=img.get_fdata()
    return img_arr

# %%
os.makedirs(TAR_PATH,exist_ok=True)
NC_PATH,MCI_PATH=TAR_PATH+"/NC",TAR_PATH+"/MCI"
os.makedirs(NC_PATH,exist_ok=True)
os.makedirs(MCI_PATH,exist_ok=True)
data=[]
for pid,dirpath in patient_dict["NC"]:
    data.append([load_t1_img(dirpath,NC_PATH+"/"+pid),0])
for pid,dirpath in patient_dict["MCI"]:
    data.append([load_t1_img(dirpath,MCI_PATH+"/"+pid),1])

# %%
df.iloc()[49]["diagonsis"]

# %%




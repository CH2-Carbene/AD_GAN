import multiprocessing
import subprocess
import os,shutil,sys
max_fail_time=5
pn_default=40
def show(s):
    print(s,flush=True)
    print(s,file=sys.stderr,flush=True)

def run_sh(cmd,name="unknown",base_dir="tmp",pname="unknown",outputs=[]):
    # tcmd=
    outputs.append(cmd)
    ret=subprocess.run(f"cd {base_dir} && {cmd} 2>&1",shell=True,stdout=subprocess.PIPE,encoding="utf")
    if ret.stdout is not None:
        outputs.append(ret.stdout)
    if ret.returncode!=0:
        show(f"{pname}.{name}: {cmd.split()[0]} Failed!")
        raise Exception(f"{name} Error!")
    show(f"{pname}.{name}: {cmd.split()[0]} Successfully!")

def getdirs(rt_dir="."):
    pres={}
    plist=os.listdir(rt_dir)
    plist.sort()
    for pname in plist:
        if not os.path.isdir(pname):continue
        slist=os.listdir(os.path.join(rt_dir,pname))
        pdict={}
        for f in slist:
            parts=f.split("_")
            if "t1" in parts: pdict["t1"]={"dir":f}
            elif "dMRI" in parts:
                if "b0" in parts:
                    td={"dir":f}
                    if "AP" in parts:td["aw"]="AP"
                    elif "PA" in parts:td["aw"]="PA"
                    else: continue
                    pdict["b0"]=td
                else:
                    td={"dir":f}
                    if "AP" in parts:td["aw"]="AP"
                    elif "PA" in parts:td["aw"]="PA"
                    else: continue
                    pdict["dti"]=td
        if "t1" in pdict or ("dti" in pdict and "b0" in pdict):
            pres[pname]=pdict

    finished_list=os.listdir("./result")
    for finished_name in finished_list:
        pres.pop(finished_name)
    return pres

def make_one_T1(pname,pdict,outputs):
    show(f"Working with {pname} to T1_prep:")
    
    TMP=f"{pname}/tmp"
    
    sh=lambda cmd,name="unknown",base_dir="tmp":run_sh(cmd,name=name,base_dir=os.path.join(pname,base_dir),pname=pname,outputs=outputs)

    if os.path.exists(os.path.join(TMP,"bet.nii.gz")):
        print(f"{pname} T1 already finished, passed!",file=sys.stderr)
        return
    
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    t1fp=os.path.join(pdict["t1"]["dir"])
    if t1fp.endswith(".tar.gz"):
        sh(f"tar -xzf {t1fp} -C ./tmp/",base_dir=".")
        t1fp=t1fp[:-7]
        sh(f"dcm2niix -b y -z y -x n -t n -m n -f t1 -o . -s n -v n {t1fp}",name="1_dcm2nii_t1")
    else:
        sh(f"dcm2niix -b y -z y -x n -t n -m n -f t1 -o . -s n -v n ../{t1fp}",name="1_dcm2nii_t1")
    
    sh(f"fslreorient2std t1 t1",name="2_reorient_t1")
    sh(f"DenoiseImage -i t1.nii.gz -n Gaussian -o t1_denoise.nii.gz -s 1",name="3_denoise_t1")
    sh(f"N4BiasFieldCorrection -d 3 --input-image t1_denoise.nii.gz --output t1_n4correct.nii.gz",name="4_N4correction")
    sh(f"flirt -in t1_n4correct.nii.gz -ref ~/data/template/MNI152_T1_0.8mm.nii.gz -out t1_ACPC.nii.gz -omat t1_ACPC.mat -dof 6",name="5_t1_to_acpc")
    sh(f"bet t1_ACPC.nii.gz t1_ACPC_brain.nii.gz -f 0.50 -R -s",name="6_t1_bet")

    os.makedirs(f"result/{pname}",exist_ok=True)
    sh(f"cp -t result/{pname} {TMP}/t1_n4correct.nii.gz {TMP}/t1_ACPC.nii.gz {TMP}/t1_ACPC_brain.nii.gz {TMP}/t1_ACPC.mat","getresult",base_dir="..")
    sh(f"mv {TMP} checkpoints/{pname}","getresult",base_dir="..")

def run_make_T1(pname,pdict):
    outputs=[]
    show(f"{pname} process start...")
    for i in range(max_fail_time):
        try:
            make_one_T1(pname,pdict,outputs)
            show(f"handle {pname} successfully!")
            show("".join(outputs))
            return
        except Exception as e:
            # print(e)
            show(f"handle {pname} Error: {e}{', retrying...' if i<4 else ', failed.'}")

    show("".join(outputs))
    raise Exception(pname,f"{pname} make T1 failed.")
    
if __name__=="__main__":

    try:
        pn=int(sys.argv[1])
    except:
        pn=pn_default
    show(f"processor_num: {(pn)}")

    os.makedirs("checkpoints",exist_ok=True)
    os.makedirs("result",exist_ok=True)
    
    # patients={
    #     "CLW_pilot-cbcp-016_103221":{
    #         "t1":{"dir":"t1_iso0.8_P2_NIF_301"},
    #         "dti":{"dir":"dMRI_1.5mm_6shells_AP_SaveBySlc_3201","aw":"AP"},
    #         "b0":{"dir":"dMRI_1.5mm_b0_PA_SaveBySlc_3301","aw":"PA"},
    #     }
    # }
    patients=getdirs()
    print(f"Get unfinished patients dict:{patients}",file=sys.stderr)
    fail_set=set()
    # raise Exception("114!")
    # exit()
    mulpool_tp = multiprocessing.Pool(processes=pn)
    for pname,pdict in patients.items():
        mulpool_tp.apply_async(run_make_T1,args=(pname,pdict,),error_callback=lambda e:fail_set.add(e.args[0]))
    # pname,pdict=list(patients.items())[0]
    mulpool_tp.close()
    mulpool_tp.join()


    print('Process finished!',file=sys.stderr)
    print(f"Fail ID:\n{' '.join(fail_set)}",file=sys.stderr)
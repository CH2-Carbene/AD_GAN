import multiprocessing
import subprocess
import os,shutil,sys
max_fail_time=1
pn_default=40

DIR=f"/hpc/data/home/bme/v-gujch/work/AD_GAN/datasets/Zhongshan_prep"
# DIR=f"datasets/Zhongshan_prep"
INDIR=f"{DIR}/paired_t2"
OUTDIR=f"{DIR}/avg"
TEM=f"/hpc/data/home/bme/v-gujch/template/MNI152_T1_0.8mm_brain.nii.gz"
GTEM_OUT=f"{DIR}/groupTemp.nii.gz"

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

def make_one_mask(sub,outputs):
    show(f"Working with {sub} to T1_prep:")
    
    TMP=f"{sub}/tmp"
    
    sh=lambda cmd,name="unknown",base_dir="tmp":run_sh(cmd,name=name,base_dir=".",pname=sub,outputs=outputs)

    # sh(rf'printf "processing $sub...\n"')
    fullsub=f"{INDIR}/{sub}"
    outsub=f"{OUTDIR}/{sub}"
    sh(f'mkdir -p {outsub}')
    sh(f"flirt -in {fullsub}/T1 -ref {TEM} -out {outsub}/T1_acpc_9dof -cost normmi -interp trilinear -dof 9")
    sh(f"fslmaths {outsub}/T1_acpc_9dof -inm 1 {outsub}/dof_norm")

def run_make_mask(sub):
    outputs=[]
    show(f"{sub} process start...",outputs)
    for i in range(max_fail_time):
        try:
            make_one_mask(sub,outputs)
            show(f"handle {sub} successfully!")
            show("".join(outputs))
            return
        except Exception as e:
            # print(e)
            show(f"handle {sub} Error: {e}{', retrying...' if i<4 else ', failed.'}")

    show("".join(outputs))
    raise Exception(sub,f"{sub} make T1 failed.")
    
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
    # patients=getdirs()
    # print(f"Get unfinished patients dict:{patients}",file=sys.stderr)

    fail_set=set()
    # raise Exception("114!")
    # exit()
    mulpool_tp = multiprocessing.Pool(processes=pn)
    for sub in os.listdir(INDIR):
        mulpool_tp.apply_async(run_make_mask,args=(sub,),error_callback=lambda e:fail_set.add(e.args[0]))
    # pname,pdict=list(patients.items())[0]
    mulpool_tp.close()
    mulpool_tp.join()

    for sub in os.listdir(INDIR):
        pass
    print('Process finished!',file=sys.stderr)
    print(f"Fail ID:\n{' '.join(fail_set)}",file=sys.stderr)
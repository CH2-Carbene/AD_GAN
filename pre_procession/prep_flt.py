import os,sys,shutil
import subprocess
import multiprocessing
pn_default=40
max_fail_time=5

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

def make_one_flt(pname,pdict,pi,outputs):
    show(f"Working with {pname} to T1_prep:")
    
    TMP=f"flt/{pi}"
    OUT=f"paired/{pi}"
    T1DIR=pdict["t1"]
    DTIDIR=pdict["dti"]

    sh=lambda cmd,name="unknown",base_dir=TMP:run_sh(cmd,name=name,base_dir=base_dir,pname=pname,outputs=outputs)

    if os.path.exists(os.path.join(OUT,"FA.nii.gz")):
        print(f"{pname} flt already finished, passed!",file=sys.stderr)
        return
    
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    
    sh(f"cp -t {TMP} {T1DIR}/* {DTIDIR}/*",name="0_getfile",base_dir=".")
    sh(f"mv t1_n4correct.nii.gz t1_ori.nii.gz",name="0_getfile")
    sh(f"mv b0_corrected_Tmean.nii.gz b0.nii.gz",name="0_getfile")
    
    sh(f"bet t1_ori.nii.gz t1_ori_brain.nii.gz -R -f 0.3 -g 0 -m",name="1_bet_T1")
    sh(f"bet b0.nii.gz b0_brain.nii.gz -R -f 0.2 -g 0 -m",name="1_bet_b0")
    sh(f"epi_reg --epi=b0_brain.nii.gz --t1=t1_ori.nii.gz --t1brain=t1_ori_brain.nii.gz --echospacing=0.00068 --out=b0_2_t1",name="2_epi_reg")

    sh(f"flirt -in t1_ori_brain.nii.gz -ref ~/template/MNI152_T1_0.8mm_brain.nii.gz -out T1 -omat t1_ACPC.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 9",name="3_flirt_ACPC")
    sh(f"convert_xfm -concat t1_ACPC.mat b0_2_t1.mat -omat b0_ACPC.mat",name="4_convert_xfm")
    sh(f"flirt -in b0 -ref T1 -out b0_ACPC -applyxfm -init b0_ACPC.mat -interp trilinear",name="5_apply_flirt")
    sh(f"flirt -in dti_FA -ref T1 -out FA -applyxfm -init b0_ACPC.mat -interp trilinear",name="5_apply_flirt")

    sh(f"mv t1_ACPC_brain.nii.gz T1.nii.gz",name="2_get_T1")

    # sh(f"flirt -in b0 -ref t1_ori -omat m1",name="flirt1")
    # sh(f"flirt -in t1_ori -ref ~/template/MNI152_T1_0.8mm.nii.gz -out T1 -omat b0_ACPC1.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 9",name="1_flirt_ACPC")
    # sh(f"flirt -in b0 -ref t1_ori -omat b0_ACPC2.mat -bins 256 -cost normmi -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 9",name="2_flirt_T1")
    # sh(f"convert_xfm -concat b0_ACPC1.mat -omat b0_ACPC.mat b0_ACPC2.mat",name="3_xfm_concat")
    # sh(f"flirt -in b0 -ref ~/template/MNI152_T1_0.8mm.nii.gz -out b0_ACPC -applyxfm -init b0_ACPC.mat -interp trilinear",name="4_apply_xfm_b0")

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    sh(f"cp {TMP}/T1.nii.gz {OUT}/T1.nii.gz",name="end_copyfile",base_dir=".")
    sh(f"cp {TMP}/FA.nii.gz {OUT}/FA.nii.gz",name="end_copyfile",base_dir=".")
    # t1fp=os.path.join(pdict["t1"]["dir"])
    # if t1fp.endswith(".tar.gz"):
    #     sh(f"tar -xzf {t1fp} -C ./tmp/",base_dir=".")
    #     t1fp=t1fp[:-7]
    #     sh(f"dcm2niix -b y -z y -x n -t n -m n -f t1 -o . -s n -v n {t1fp}",name="1_dcm2nii_t1")
    # else:
    #     sh(f"dcm2niix -b y -z y -x n -t n -m n -f t1 -o . -s n -v n ../{t1fp}",name="1_dcm2nii_t1")
    
    # sh(f"fslreorient2std t1 t1",name="2_reorient_t1")
    # sh(f"DenoiseImage -i t1.nii.gz -n Gaussian -o t1_denoise.nii.gz -s 1",name="3_denoise_t1")
    # sh(f"N4BiasFieldCorrection -d 3 --input-image t1_denoise.nii.gz --output t1_n4correct.nii.gz",name="4_N4correction")
    # sh(f"flirt -in t1_n4correct.nii.gz -ref ~/data/template/MNI152_T1_0.8mm.nii.gz -out t1_ACPC.nii.gz -omat t1_ACPC.mat -dof 6",name="5_t1_to_acpc")
    # sh(f"bet t1_ACPC.nii.gz t1_ACPC_brain.nii.gz -f 0.50 -R -s",name="6_t1_bet")

    # os.makedirs(f"result/{pname}",exist_ok=True)
    # sh(f"cp -t result/{pname} {TMP}/t1_n4correct.nii.gz {TMP}/t1_ACPC.nii.gz {TMP}/t1_ACPC_brain.nii.gz {TMP}/t1_ACPC.mat","getresult",base_dir="..")
    # sh(f"mv {TMP} checkpoints/{pname}","getresult",base_dir="..")

def run_make_flt(pname,pdict,pi):
    outputs=[]
    show(f"{pname} process start...",outputs)
    for i in range(max_fail_time):
        try:
            make_one_flt(pname,pdict,pi,outputs)
            show(f"handle {pname} successfully!")
            show("".join(outputs))
            return
        except Exception as e:
            # print(e)
            show(f"handle {pname} Error: {e}{', retrying...' if i<4 else ', failed.'}")

    show("".join(outputs))
    raise Exception(pname,f"{pname} make T1 failed.")

def get_ptdict():
    ptdict={}
    for desti in os.listdir("t1"):
        if not os.path.isdir(f"t1/{desti}"):continue
        for pati in os.listdir(f"t1/{desti}"):
            if os.path.isdir(f"t1/{desti}/{pati}") and pati!="result" and pati!="checkpoints":
                pid=f"{desti}/{pati}"
                if pid not in ptdict:
                    ptdict[pid]={}

                t1dir=f"t1/{desti}/result/{pati}"
                if os.path.isdir(t1dir):
                    ptdict[pid]["t1"]=t1dir
    
    for desti in os.listdir("dti"):
        if not os.path.isdir(f"dti/{desti}"):continue
        for pati in os.listdir(f"dti/{desti}"):
            if os.path.isdir(f"dti/{desti}/{pati}") and pati!="result" and pati!="checkpoints":
                pid=f"{desti}/{pati}"
                if pid not in ptdict:
                    ptdict[pid]={}

                dtidir=f"dti/{desti}/result/{pati}"
                if os.path.isdir(dtidir):
                    ptdict[pid]["dti"]=dtidir
    return ptdict

if __name__=='__main__':
    try:
        pn=int(sys.argv[1])
    except:
        pn=pn_default
    show(f"processor_num: {(pn)}")
    os.makedirs("flt",exist_ok=True)
    os.makedirs("paired",exist_ok=True)

    ptdict=get_ptdict()
    show(ptdict)

    fail_set=set()
    mulpool_flt = multiprocessing.Pool(processes=pn)

    finished_set=set(os.listdir("./paired"))
    not1,nodti=[],[]
    for i,pid in enumerate(sorted(ptdict.keys())):
        if str(i) in finished_set:
            show(f"{i}_{pid} has finished, passed.")
            continue
        ptdir=ptdict[pid]
        t1dir=ptdir.get("t1")
        dtidir=ptdir.get("dti")
        if t1dir is None:
            show(f"{pid} have no T1")
            not1.append(pid)
            continue
        if dtidir is None:
            show(f"{pid} have no dti")
            nodti.append(pid)
            continue
        mulpool_flt.apply_async(run_make_flt,args=(pid,ptdir,i,),error_callback=lambda e:fail_set.add(e.args[0]))
    mulpool_flt.close()
    mulpool_flt.join()
    show(f"No t1:{not1}")
    show(f"No dti:{nodti}")
    
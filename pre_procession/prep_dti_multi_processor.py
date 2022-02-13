import multiprocessing
import subprocess
import os,shutil,sys
max_fail_time=5
pn_tp_default,pn_eddy_default=40,16

def show(s):
    print(s)
    print(s,file=sys.stderr)

def run_sh(cmd,name="unknown",base_dir="tmp",pname="unknown",outputs=[]):
    # tcmd=
    print(cmd)
    ret=subprocess.run(f"cd {base_dir} && {cmd} 2>&1",shell=True)
    outputs.append(ret.stdout.decode("utf-8"))
    if ret.returncode!=0:
        print(f"{pname}.{name}: {cmd.split()[0]} Failed!")
        raise Exception(f"{name} Error!")
    print(f"{pname}.{name}: {cmd.split()[0]} Successfully!")
    

# print(a)
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
# print(getdirs())
def make_one_dti_topup(pname,pdict,outputs):
    show(f"Working with {pname} to topup:")
    
    TMP=f"{pname}/tmp"
    
    sh=lambda cmd,name="unknown",base_dir="tmp":run_sh(cmd,name=name,base_dir=os.path.join(pname,base_dir),pname=pname,outputs=outputs)

    if os.path.exists(os.path.join(TMP,"b0_brain_mask.nii.gz")):
        print(f"{pname} topup already finished, passed!",file=sys.stderr)
        return
    
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    dtifp=os.path.join(pdict["dti"]["dir"])
    dtiaw=pdict["dti"]["aw"]
    b0fp=os.path.join(pdict["b0"]["dir"])
    b0aw=pdict["b0"]["aw"]

    if dtifp.endswith(".tar.gz"):
        sh(f"tar -xzf {dtifp} -C ./tmp/",base_dir=".")
        dtifp=dtifp[:-7]
        sh(f"dcm2niix -b y -z y -x n -t n -m n -f dti -o . -s n -v n {dtifp}",name="1_dcm2nii_dti")
    else:
        sh(f"dcm2niix -b y -z y -x n -t n -m n -f dti -o . -s n -v n ../{dtifp}",name="1_dcm2nii_dti")
    
    if b0fp.endswith(".tar.gz"):
        sh(f"tar -xzf {b0fp} -C ./tmp/",base_dir=".")
        b0fp=b0fp[:-7]
        sh(f"dcm2niix -b y -z y -x n -t n -m n -f b0 -o . -s n -v n {b0fp}",name="1_dcm2nii_b0")
    else:
        sh(f"dcm2niix -b y -z y -x n -t n -m n -f b0 -o . -s n -v n ../{b0fp}",name="1_dcm2nii_b0")
    # raise("dcm2niix finish!")

    sh(f"fslreorient2std dti dti",name="2_reorient_dti")
    sh(f"fslreorient2std b0 b0",name="2_reorient_b0")

    imgdict={}
    # a=os.popen('fslinfo tmp/dti').readlines()
    for i,line in enumerate(os.popen(f'fslinfo {TMP}/b0').readlines()):
        x,y=line.split()
        imgdict[x]=y
    dimx,dimy,dimz,b0dimt=map(int,[imgdict["dim1"],imgdict["dim2"],imgdict["dim3"],imgdict["dim4"]])
    dimx+=dimx%2;dimy+=dimy%2;dimz+=dimz%2
    sh(f"fslroi dti dti_roi 0 {dimx} 0 {dimy} 0 {dimz}",name="2_roi_dti")
    sh(f"fslroi b0 b0_roi 0 {dimx} 0 {dimy} 0 {dimz}",name="2_roi_b0")

    print(dimx,dimy,dimz,b0dimt)
    #TODO dwidenoise && mrdegibbs
    sh("dwidenoise b0_roi.nii.gz b0_dns1.nii.gz",name="2.5_denoise")
    sh("mrdegibbs b0_dns1.nii.gz b0_dns.nii.gz",name="2.5_denoise")
    sh("dwidenoise dti_roi.nii.gz dti_dns1.nii.gz",name="2.5_denoise")
    sh("mrdegibbs dti_dns1.nii.gz dti_dns.nii.gz",name="2.5_denoise")
    
    sh("fslchfiletype NIFTI dti_dns data","fslchfiletype")
    with open(f"{TMP}/dti.bval") as f:
        bval_arr=list(map(int,f.readline().split()))
    idlist=[]
    for i,x in enumerate(bval_arr):
        if x==0:idlist.append(i)
    os.makedirs(f"{TMP}/b0tmp",exist_ok=True)
    for id in idlist:
        sh(f"fslroi data b0tmp/{id} {id} 1",name="3_roi_merge")
    
    def getdy(aw):
        if aw=="AP": return -1
        elif aw=="PA": return 1
        else: raise Exception("unknown M direction!")
    b0list_str=" ".join(map(str,idlist))
    sh(f"fslmerge -t ../datab0.nii.gz {b0list_str}",base_dir="tmp/b0tmp",name="3_roi_merge")
    sh(f"fslmerge -t allb0.nii.gz datab0.nii.gz b0_dns.nii.gz",name="3_roi_merge")
    shutil.rmtree(f"{TMP}/b0tmp")
    # os.remove("tmp/datab0.nii.gz")
    # os.remove("tmp/b0_roi.nii.gz")


    
    with open(f"{TMP}/acqp.txt",mode="w") as f:
        for x in idlist:
            f.write(f"0 {getdy(dtiaw)} 0 0.1\n")
        for x in range(b0dimt):
            f.write(f"0 {getdy(b0aw)} 0 0.1\n")

    with open(f"{TMP}/index.txt",mode="w") as f:
        t=0
        for x in bval_arr:
            if x==0:t+=1
            f.write(f"{t} ")

    
    sh("topup --config=b02b0.cnf --datain=acqp.txt --imain=allb0.nii.gz --out=tpbase --iout=b0_corrected.nii.gz --fout=b0_field.nii.gz --logout=b0_topup.log -v",name="4_topup")

    
    # topup afterwards
    #TODO check topup result here!
    sh("fslmaths ./b0_corrected.nii.gz -Tmean ./b0_corrected_Tmean.nii.gz",name="4_topup_tmean")
    sh("bet b0_corrected_Tmean.nii.gz b0_brain.nii.gz -f 0.3 -g 0 -m",name="4_topup_bet")
    #TODO check bet result here!

class TNFException(BaseException):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def make_one_dti_eddy(pname,pdict,outputs):
    show(f"Working with {pname} to eddy:")
    

    TMP=f"{pname}/tmp"
    sh=lambda cmd,name="unknown",base_dir="tmp":run_sh(cmd,name=name,base_dir=os.path.join(pname,base_dir),pname=pname,outputs=outputs)

    if not os.path.exists(os.path.join(TMP,"b0_brain_mask.nii.gz")):
        raise TNFException("Topup not finished!")
        print("Warning: {pname} topup not finished, retrying...",file=sys.stderr)
    
    sh("eddy_openmp --imain=data --mask=b0_brain_mask.nii.gz --bvals=dti.bval --bvecs=dti.bvec --acqp=acqp.txt --index=index.txt --out=data_corrected --ref_scan_no=0 --ol_nstd=4 --topup=tpbase -v",name="5_eddy")

    sh("rm data.nii",name="6_rmdata")
    sh("rm dti.nii.gz",name="6_rmdata")
    
    os.makedirs(f"{TMP}/fitresult",exist_ok=True)
    sh("dtifit --data=data_corrected --out=fitresult/dti --mask=b0_brain_mask.nii.gz --bvecs=data_corrected.eddy_rotated_bvecs --bvals=dti.bval --sse --save_tensor",name="6_dtifit",base_dir="tmp")

    os.makedirs(f"result/{pname}",exist_ok=True)
    sh(f"cp {TMP}/fitresult/* result/{pname}","getresult",base_dir="..")
    sh(f"mv {TMP} checkpoints/{pname}","getresult",base_dir="..")


    
def run_make_topup(pname,pdict):
    outputs=[]
    show(f"{pname} process start...")
    for i in range(max_fail_time):
        try:
            make_one_dti_topup(pname,pdict,outputs)
            show(f"handle {pname} successfully!")
            show("".join(outputs))
            return
        except Exception as e:
            # print(e)
            show(f"handle {pname} Error: {e}{', retrying...' if i<4 else ', failed.'}")
            
    show("".join(outputs))
    raise Exception(pname,f"{pname} topup failed.")

def run_make_eddy(pname,pdict):
    outputs=[]
    show(f"{pname} process start...")
    for i in range(max_fail_time):
        try:
            make_one_dti_eddy(pname,pdict,outputs)
            show(f"handle {pname} successfully!")
            show("".join(outputs))
            return
        except TNFException as e:
            show(f"handle {pname} Error: {e}, process failed.")
            break
            # raise Exception(pname,f"{pname} eddy failed.")
        except Exception as e:
            # print(e)
            show(f"handle {pname} Error: {e}{', retrying...' if i<4 else ', failed.'}")
    show("".join(outputs))
    raise Exception(pname,f"{pname} eddy failed.")

if __name__=="__main__":

    try:
        pn_tp,pn_eddy=int(sys.argv[1],sys.argv[2])
    except:
        pn_tp,pn_eddy=pn_tp_default,pn_eddy_default
    show(f"processor_num: {(pn_tp,pn_eddy)}")

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
    mulpool_tp = multiprocessing.Pool(processes=pn_tp)
    for pname,pdict in patients.items():
        mulpool_tp.apply_async(run_make_topup,args=(pname,pdict,),error_callback=lambda e:fail_set.add(e.args[0]))
    # pname,pdict=list(patients.items())[0]
    mulpool_tp.close()
    mulpool_tp.join()

    mulpool_ed = multiprocessing.Pool(processes=pn_eddy)
    for pname,pdict in patients.items():
        mulpool_ed.apply_async(run_make_eddy,args=(pname,pdict,),error_callback=lambda e:fail_set.add(e.args[0]))
    # pname,pdict=list(patients.items())[0]
    mulpool_ed.close()
    mulpool_ed.join()

    print('Process finished!',file=sys.stderr)
    print(f"Fail ID:\n{' '.join(fail_set)}",file=sys.stderr)
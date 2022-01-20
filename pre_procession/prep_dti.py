
import os,shutil,sys
def run_sh(cmd,name="unknown",base_dir="tmp"):
    # tcmd=
    print(cmd)
    res=os.system(f"cd {base_dir} && {cmd}")
    if res!=0:
        raise Exception(f"{name} Error!")
    print(f"{name}: {cmd.split()[0]} Successfully!")
# print(a)
def getdirs(rt_dir="."):
    pres={}
    plist=os.listdir(rt_dir)
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
    return pres
# print(getdirs())
def make_one_dti(pname,pdict):
    print(pname,pdict)

    TMP=f"{pname}/tmp"
    sh=lambda cmd,name="unknown",base_dir="tmp":run_sh(cmd,name=name,base_dir=os.path.join(pname,base_dir))

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

    
    sh("fslchfiletype NIFTI dti_roi data","fslchfiletype")
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
    sh(f"fslmerge -t allb0.nii.gz datab0.nii.gz b0_roi.nii.gz",name="3_roi_merge")
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
            f.write(f"{1} ")

    
    sh("topup --config=b02b0.cnf --datain=acqp.txt --imain=allb0.nii.gz --out=tpbase --iout=b0_corrected.nii.gz --fout=b0_field.nii.gz --logout=b0_topup.log -v",name="4_topup")

    
    # topup afterwards
    #TODO check topup result here!
    sh("fslmaths ./b0_corrected.nii.gz -Tmean ./b0_corrected_Tmean.nii.gz",name="4_topup_tmean")
    sh("bet b0_corrected_Tmean.nii.gz b0_brain.nii.gz -m",name="4_topup_bet")
    #TODO check bet result here!

    
    sh("eddy_openmp --imain=data --mask=b0_brain_mask.nii.gz --bvals=dti.bval --bvecs=dti.bvec --acqp=acqp.txt --index=index.txt --out=data_corrected.nii.gz --ref_scan_no=0 --ol_nstd=4 --topup=tpbase -v",name="5_eddy")

    
    os.makedirs(f"{TMP}/fitresult",exist_ok=True)
    sh("dtifit --data=data_corrected.nii.gz --out=fitresult/dti --mask=b0_brain_mask.nii.gz --bvecs=data_corrected.eddy_rotated_bvecs --bvals=dti.bval --sse --save_tensor",name="6_dtifit",base_dir="tmp")

    
    sh(f"cp {TMP}/fitresult/* result/{pname}","getresult",base_dir="..")
    sh(f"mv {TMP} checkpoints/{pname}","getresult",base_dir="..")




if __name__=="__main__":
    
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
    print(patients)
    # raise Exception("114!")
    # exit()
    for pname,pdict in patients.items():
    # pname,pdict=list(patients.items())[0]
        print(pname,pdict)
        try:
            make_one_dti(pname,pdict)
            sys.stderr.write(f"handle {pname} successfully!\n")
        except Exception as e:
            sys.stderr.write(f"handle {pname} Error: {e}\n")
        
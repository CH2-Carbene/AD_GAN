import multiprocessing
import os,shutil,sys
from tqdm import tqdm

def checko(path,file):
    imgdict={}
    # a=os.popen('fslinfo tmp/dti').readlines()
    for i,line in enumerate(os.popen(f'fslhd {path}/{file}').readlines()):
        o=line.split()
        # print(len)
        if len(o)>1:
            imgdict[o[0]]=o[1:]
    # print(imgdict)
    # s=os.popen(f'fslorient -getorient {path}/{file}').readlines()[0].split()[0]
    s="".join([imgdict["sform_xorient"][0][0],imgdict["sform_yorient"][0][0],imgdict["sform_zorient"][0][0]])
    if s!="RPI":
        print(f"{path}/{file}: Not RPI!")
        print(s)
        # input()
    else:
        # print(f"{path}/{file}: is RPI!")
        pass
    # print(114514)

mulpool = multiprocessing.Pool(processes=44)
for path,dirs,files in os.walk("."):
    for file in files:
        if file=="dti_roi.nii.gz" or file=="b0_roi.nii.gz":
            # print(path,file)
            # checko(path,file)
            mulpool.apply_async(checko,args=(path,file))
            # print(file)
mulpool.close()
mulpool.join()
print("finish!")
    # print(path,dir,file)

    # print(os.path.join(path,dir,file))
# os.popen

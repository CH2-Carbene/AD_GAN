from scipy import stats
import os
import numpy as np

PATH="/hpc/data/home/bme/v-gujch/bme_data/ZS_t1_full"
# PATH=r"C:\Users\CH2\Documents\datasets\ZS_t1_full"
PATCH=PATH+"/patches/"
TT_DIR=PATH+"/TT_feq/"

def calc_tt(nowmoda="T1"):
    lmci=[]
    lnc=[]
    for pid in os.listdir(PATCH):
        npf=np.load(os.path.join(PATCH,pid,"patch_13.npz"))
        print(npf["label"])
        if npf["label"]=="MCI":
            lmci.append(npf[nowmoda])
        else:
            lnc.append(npf[nowmoda])
    # levene = stats.levene(list1, list2, center='median')
    # print('w-value=%6.4f,p-value=%6.4f' % levene)
    st,p=stats.stats.ttest_ind(lmci, lnc,axis=0, equal_var=False)
    os.makedirs(TT_DIR,exist_ok=True)
    np.save(TT_DIR+nowmoda+"_st",st)
    np.save(TT_DIR+nowmoda+"_p",p)
    return st,p
if __name__ == '__main__':
    cklist=["T1","delta1","delta2","delta3"]
    for mod in cklist:
        calc_tt(mod)

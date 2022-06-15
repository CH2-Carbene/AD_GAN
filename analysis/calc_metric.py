import numpy as np
def calc_metric(real,fak):
    try:
        from skimage.metrics import structural_similarity as SSIM
        from skimage.metrics import peak_signal_noise_ratio as PSNR
        from skimage.metrics import mean_squared_error as MSE
        from skimage.metrics import normalized_mutual_information as NMI

        real[real>1]=1
        fak[fak>1]=1
        d={"MSE":MSE(real,fak),"SSIM":SSIM(real,fak),"PSNR":PSNR(real,fak),"NMI":NMI(real,fak)}
        return d
    except:
        return {}

def eval_result(self,patch_list):
    pMCI={"MSE":0,"SSIM":0,"PSNR":0,"NMI":0}
    pNC={"MSE":0,"SSIM":0,"PSNR":0,"NMI":0}
    num=0
    
    for npfile in patch_list:
        z=np.load(npfile)
        z["T1"],z["Fake_"]
        z[""]
        dA=calc_metric(imgA[0,...,0].numpy())
        for x,y in dA.items():
            pA[x]+=y
        for x,y in dB.items():
            pB[x]+=y
        num+=1
    print(res)
    return resA,resB
    
PATH="/hpc/data/home/bme/v-gujch/bme_data/ZS_t1_full/patches"

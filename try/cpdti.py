import os
import sys
import shutil
from tqdm import tqdm
# source = 'current/test/test.py'
# target = '/prod/new'

# assert not os.path.isabs(source)
# target = os.path.join(target, os.path.dirname(source))
tgdir = "/public_bme/data/cbcp_dwi"
# create the folders if not already exists
# os.makedirs(target,exist_ok=True)
with open("use_list.txt") as f:
    for s in tqdm(f.readlines()):
        # s=f.readline()
        s = s[:-1]
        sval = s[:-7]+".bvals"
        svec = s[:-7]+".bvecs"
        ps = s[18:]

        ps = os.path.join(tgdir, os.path.dirname(ps))
        print(ps, flush=True)
        os.makedirs(ps, exist_ok=True)
        try:
            shutil.copy(s, ps)
            if os.path.exists(sval):
                shutil.copy(sval, ps)
            if os.path.exists(svec):
                shutil.copy(svec, ps)
        except IOError as e:
            print("Unable to copy file. %s" % e)
        except:
            print("Unexpected error:", sys.exc_info())
        # shutil.copy(s,tgdir)
        # break
# adding exception handling
# try:
#    shutil.copy(source, target)
# except IOError as e:
#    print("Unable to copy file. %s" % e)
# except:
#    print("Unexpected error:", sys.exc_info())

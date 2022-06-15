### TODO
#!/usr/bin bash

# /hpc/data/home/bme/yubw/Output/subjects/${pname}/ses-M00/t1/freesurfer_cross_sectional/${pname}_ses-M00/mri/orig/


# SUBJECT_DIR=/hpc/data/home/bme/yubw/Output/subjects
# file="name.txt"

# TAR_DIR=/

SUBJECT_DIR=/hpc/data/home/bme/yubw/Output/subjects
for pname in $(ls ${SUBJECT_DIR})
do
echo copying ${pname} ...
cp /hpc/data/home/bme/yubw/Output/subjects/${pname}/ses-M00/t1/freesurfer_cross_sectional/${pname}_ses-M00/mri/orig/001.mgz /public_bme/data/gujch/ZS_t1_full/${pname}.mgz
# echo "visit beautiful $state"
done
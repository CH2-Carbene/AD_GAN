#!/bin/bash
#SBATCH -J make_mask
#SBATCH -p bme_cpu
#SBATCH --time=300
#SBATCH --mem=16G
#SBATCH -N 1
#SBATCH -n 16 

#SBATCH -o %j_stdout.log
#SBATCH -e %j_stderr.log

DIR="$HOME/work/AD_GAN/datasets/Zhongshan_prep"
INDIR="$DIR/paired_t2"
OUTDIR="$DIR/avg"
TEM="$HOME/template/MNI152_T1_0.8mm_brain.nii.gz"
GTEM_OUT="$DIR/groupTemp.nii.gz"

# for sub in $(ls $INDIR);do

#     printf "processing $sub...\n"
#     fullsub=$INDIR/$sub
#     outsub=$OUTDIR/$sub
#     mkdir -p $outsub
#     flirt -in $fullsub/T1 -ref $TEM -out $outsub/T1_acpc_9dof -cost normmi -interp trilinear -dof 9
#     fslmaths $outsub/T1_acpc_9dof -inm 1 $outsub/dof_norm
# done

for sub in $(ls $OUTDIR);
do
    outsub=$OUTDIR/$sub
    if [ -z $OUTEXT ]
    then
        cp $outsub/dof_norm.nii.gz $GTEM_OUT
        OUTEXT="1"
    else
        fslmaths $GTEM_OUT -add $outsub/dof_norm $GTEM_OUT
    fi
done
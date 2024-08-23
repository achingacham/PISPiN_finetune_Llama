#!/usr/bin/env bash
###!/bin/bash


# run setup
source /nethome/achingacham/HTCondor_prep/scripts/setup.sh


# run misc. stuff
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# conda env
HOME="/nethome/achingacham/"
ls /data/users/achingacham/anaconda3/etc/profile.d/conda.sh -lah
source /data/users/achingacham/anaconda3/etc/profile.d/conda.sh
conda activate llama #pytorch_1_6_clone

echo "Inside conda"

echo $HOSTNAME
which python
#python -m pip list

#prompt-prefix #[for the a listening condition with NT at NL]
pPrefix=$'For the given input text, generate an acoustically better intelligible paraphrase with 10-12 words\n###Input:\n'

#promp-infix
pInfix=$'\n###Response:\n'

### FINETUNE (with PiN - existing dataset file)
#python /nethome/achingacham/PycharmProjects/LLaMA/scripts/sft_llama.py \
#--dataSetFile "/projects/SFB_A4/A4-ParaphrasesinNoise/PiN.tsv_NoiseID2_lowPhER_to_highPhER" \
#-dsFile_iCol "low-SI-para" \
#-dsFile_oCol "high-SI-para" \
#-pPrefix "$pPrefix" \
#-pInfix "$pInfix"



### FINETUNE (with SWDA - using an existing DATASET file)
#dSFile_small: babble, SNR -5 - "/projects/SFB_A4/llama-2/llama-ft-models/09-18-2023_03:50:51_SWDA_dataset/dataset.tsv" \
#dSFile_small: white, SNR -5 - "/projects/SFB_A4/llama-2/llama-ft-models/09-23-2023_18:36:44_SWDA_dataset/all_dataset.tsv" \

#dSFile_large: babble, SNR -5 - "/projects/SFB_A4/llama-2/llama-ft-models/09-30-2023_00:50:02_SWDA_babble/all_dataset.tsv" \
#dSFile_large: white, SNR -5 - "/projects/SFB_A4/llama-2/llama-ft-models/09-30-2023_00:43:36_SWDA_white/all_dataset.tsv"

python /nethome/achingacham/PycharmProjects/LLaMA/scripts/sft_llama.py \
--dataSetFile "/projects/SFB_A4/llama-2/llama-ft-models/09-30-2023_00:43:36_SWDA_white/all_dataset.tsv" \
-bSize 8 \
-pPrefix "$pPrefix" \
-pInfix "$pInfix" \
--thresholdValue 1.1

### FINETUNE (with SWDA - creating a new DATA file)
#python /nethome/achingacham/PycharmProjects/LLaMA/scripts/sft_llama.py \
#--dataFile "/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/train_SWDA_00-05-text2speech_all.tsv" \
#-pPrefix "$pPrefix" \
#-pInfix "$pInfix" \
#--thresholdValue 1.0 \
#--noiseDirectory "/projects/SFB_A4/AudioRepo/noise_babble/"

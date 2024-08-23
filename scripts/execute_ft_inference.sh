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


### INFERENCE

#prompt-prefix #[for the a listening condition with NT at NL]
promptPrefix=$'For the given input text, generate an acoustically better intelligible paraphrase with 10-12 words\n###Input:\n'

#promp-infix
promptInfix=$'\n###Response:\n'


# ft_PiN (train size: 300)
#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/09-26-2023_18:21:04_PiN/checkpoint-300"

#SWDA - small
#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/09-19-2023_21:45:16" # ft_SWDA_Babble_SNR-5
#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/09-22-2023_06:19:30/checkpoint-972"
#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/09-23-2023_22:00:53/checkpoint-1166"  #SWDA_gt_1.1_white (trainsize: 2K)

#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/09-26-2023_18:16:44/checkpoint-2331"  # SWDA(15K)-gt 1.1
#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/09-27-2023_00:48:19/checkpoint-1068"  # SWDA(4K)-gt 1.2

#babble - SWDA_large
#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/09-30-2023_04:28:08_SWDA_l_babble/checkpoint-9879"  #pwr-STOI ge 1.0
#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/10-02-2023_23:53:02_SWDA_l_babble/checkpoint-3774" #pwr-STOI ge 1.1
sft_model="/projects/SFB_A4/llama-2/llama-ft-models/10-02-2023_23:49:30_SWDA_l_babble/checkpoint-2336" #pwr-STOI ge 1.2

#white - SWDA_large
#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/09-30-2023_04:34:54_SWDA_l_white/checkpoint-11481" #pwr-STOI ge 1.0
#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/10-02-2023_23:54:49_SWDA_l_white/checkpoint-3366" #pwr-STOI ge 1.1
#sft_model="/projects/SFB_A4/llama-2/llama-ft-models/10-02-2023_23:59:40_SWDA_l_white/checkpoint-606" #pwr-STOI ge 1.2


python /nethome/achingacham/PycharmProjects/LLaMA/scripts/inference_finetuned_llama.py  \
-pPrefix "$promptPrefix" \
-pInfix "$promptInfix" \
-mPath "$sft_model" \
#--disableAdapterLayers #for the base model


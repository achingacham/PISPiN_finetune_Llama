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

### INFERENCE

#prompt-prefix #[for the a listening condition with NT at NL]
promptPrefix=$'For the given input text, generate an acoustically better intelligible paraphrase with 10-12 words\n###Input:'

#promp-infix
promptInfix=$'\n###Response:'

# script arguments:

python /nethome/achingacham/PycharmProjects/LLaMA/scripts/inference_llama.py \
        -pPrefix "$promptPrefix" \
        -pInfix "$promptInfix" \
        # -iFile "/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv" \
        # -mName "meta-llama/Llama-2-7b-chat-hf"


# Extras:
# if split_row and split_cols are not created:

# python3 $HOME/PycharmProjects/chatGPT/scripts/split_output.py -oFile $output_file
# python3 $HOME/PycharmProjects/LLaMA/scripts/llama_hf.py
# python3 $HOME/PycharmProjects/LLaMA/scripts/cublas_tshoot.py
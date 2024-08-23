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


## run python / bash script 

#script_dir=/projects/SFB_A4/llama-2/llama/
#MP=1 # llama-2-7B model
#torchrun --nproc_per_node $MP "$script_dir/example_text_completion.py" \
#    --ckpt_dir "$script_dir/llama-2-7b/" \
#    --tokenizer_path "$script_dir/tokenizer.model" \
#    --max_seq_len 128 --max_batch_size 4


#python /nethome/achingacham/PycharmProjects/LLaMA/scripts/finetune_llama.py

#python /nethome/achingacham/PycharmProjects/LLaMA/scripts/infer_finetuned_llama.py


####PROMPT for in-context learning####
#kw_split="\n\nAnswer:\n"
#user_prompt=$"Look at the list of examples of a sentence and its intelligible paraphrase:
#1. Apart from college, what are your other long - term objectives ?  =>  what other long range goals do you have besides college ?
#2. Are you acquainted with that ? I'm not aware.  =>  I don't know you're familiar with that or not.
#3. Feeling stuck between two generations can be quite stressful.  =>  you feel squeezed in the middle of having both generations ,
#4. What is the typical timeframe for your contribution to become fully vested ?  =>  how long does it take for your contribution to vest ?
#5. In the past , I had access , but currently , I don't.  =>  I don't have access either. Although , I did at one time
#Similarly, generate an intelligible paraphrase  for the sentence:
#"

####PROMPT to generate a list of paraphrases####
#kw_split="\n\nAnswer:\n"; user_prompt="Generate a numbered list of 6 simple, intelligible, spoken-styled paraphrases with 10 to 12 words for the following input sentence: "

####PROMPT to generate a single paraphrase####
kw_split="\n\nAnswer:\n\""; user_prompt="Generate without any explanation. Generate a simple, intelligible, spoken-styled paraphrases with 10 to 12 words for the following input sentence: "

file_path="/projects/SFB_A4/llama-2/llama-data/SWDA-PiSPIN/sample_input_text.txt" #$1
echo "Input_file: $file_path"
##OR iFile /projects/SFB_A4/llama-2/llama-data/SWDA-PiSPIN/input_text_aa.txt \

#python /nethome/achingacham/PycharmProjects/LLaMA/scripts/inference_llama.py \
#        -iFile "$file_path" \
#        -hRow 0 \
#        -iCol 'input_text' \
#        -uPrompt "$user_prompt" \
#        --kw_split_prompt_response "$kw_split" # -mName "meta-llama/Llama-2-13b-chat-hf"



### INFERENCE

#p_c
#instruction="For a noisy listening environment with babble noise at SNR -5, generate a simple, intelligible, and spoken-styled paraphrase with 10-12 words, for the following input sentence: "
#python /nethome/achingacham/PycharmProjects/LLaMA/scripts/infer_finetuned_llama.py  \
#      -pPrefix "$instruction"


### FINETUNE
python /nethome/achingacham/PycharmProjects/LLaMA/scripts/sft_llama.py
#python /nethome/achingacham/PycharmProjects/LLaMA/scripts/finetune_llama.py


# if split_row and split_cols are not created:
# python3 $HOME/PycharmProjects/chatGPT/scripts/split_output.py -oFile $output_file

#python3 $HOME/PycharmProjects/LLaMA/scripts/llama_hf.py

#python3 $HOME/PycharmProjects/LLaMA/scripts/cublas_tshoot.py


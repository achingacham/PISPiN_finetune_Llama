# base code credits: https://huggingface.co/docs/trl/sft_trainer
# !pip install -q huggingface_hub
# !pip install -q -U trl transformers accelerate peft
# !pip install -q -U datasets bitsandbytes einops wandb

# Uncomment to install new features that support latest models like Llama 2
# !pip install git+https://github.com/huggingface/peft.git
# !pip install git+https://github.com/huggingface/transformers.git

# When prompted, paste the HF access token you created earlier.
from huggingface_hub import notebook_login
#notebook_login()

import os
import ipdb
import time
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from math import factorial
from itertools import combinations, permutations
from datasets import load_dataset, Dataset

from sfba4.utils import plotGraphs as pG
from sfba4.utils import create_mixed_audio_file as mix_audio
from sent_comp import get_features

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


np.random.seed(42)



def make_pair_rows(gdf,
                   pair_id_col='pair_idx',
                   **kwargs):

    n_items = len(gdf)
    # ipdb.set_trace()
    # row_indices = combinations(range(n_items), 2)  # n! / ((n-r)! * r!)
    row_indices = permutations(range(n_items), 2)  # n!/(n-r)!

    list_gdf_pairs = []

    for idx, row_id in enumerate(row_indices):
        gdf_pair = gdf.iloc[list(row_id)]
        gdf_pair[pair_id_col] = idx
        list_gdf_pairs.append(gdf_pair)

    gdf_paired_items = pd.concat(list_gdf_pairs)
    return gdf_paired_items


def mix_noise_find_stoi(clean_file=None,
                        noise_type_file=None,
                        noise_level=None,
                        noise_type_file_start=None):


    args = mix_audio.get_args()
    args.mixNoise = True

    if clean_file is not None:
        args.clean_file = clean_file

    if noise_type_file is not None:
        args.noise_file = noise_type_file

    if noise_level is not None:
        args.snr = noise_level

    if noise_type_file_start is not None:
        args.max_mix_length = noise_type_file_start

    #print(args)

    mix_audio.perform_additive_noise_mixing(args)
    stoi = get_features.get_STOI(args.clean_file, args.output_mixed_file)

    return stoi


def add_noise_cols(gdf,
                   group_id_col="paraphrases_id",
                   noise_levels=None,
                   noise_types=None,
                   expand=False,
                   pw_STOI_ratio_threshold=1.0,
                   **kwargs):

    if expand == True:
        list_gdf = []
        for each_noise_type in noise_types:
            for each_noise_level in noise_levels:
                gdf['noise_type'] = each_noise_type
                gdf['noise_level'] = each_noise_level
                list_gdf.append(gdf)

        gdf = pd.concat(list_gdf)
    else:
        gdf['noise_type'] = np.random.choice(noise_types)
        gdf['noise_level'] = np.random.choice(noise_levels)


    #ipdb.set_trace()

    # 2.mix noise for each row (with a pair_id) and calculate the STOI
    #find stoi for all permutations of pairs and filter the best among them.
    #print("Start time of stoi calculation per group:", time.ctime())
    gdf['STOI'] = gdf.apply(lambda row: mix_noise_find_stoi(clean_file=row['clean_utt_path'],
                                                            noise_type_file=row['noise_type'],
                                                            noise_level=row['noise_level'],
                                                            noise_type_file_start=row[group_id_col]), axis=1)

    #print("End time of stoi calculation per group:", time.ctime())
    #ipdb.set_trace()
    index_mapping = {i: idx for i, idx in enumerate(gdf.index)}

    stoi_values = gdf['STOI'].to_numpy()
    pw_ratios_stoi = np.matmul(np.expand_dims(1/stoi_values, 1), np.expand_dims(stoi_values, 0))

    # assign zero to all diagonal items (to ignore them):
    for i in range(len(stoi_values)):
        pw_ratios_stoi[i][i] = 0

    row_ind, col_ind = linear_sum_assignment(-1 * pw_ratios_stoi)  # to optimize the function by maximizing

    low_stoi_paraphrases = []
    high_stoi_paraphrases = []

    low_stoi_values = []
    high_stoi_values = []

    ratio_stoi_values = []

    for i, j in zip(row_ind, col_ind):
        if pw_ratios_stoi[i, j] > pw_STOI_ratio_threshold:
            low_stoi_paraphrases.append(gdf.iloc[i]['text'])
            high_stoi_paraphrases.append(gdf.iloc[j]['text'])

            low_stoi_values.append(gdf.iloc[i]['STOI'])
            high_stoi_values.append(gdf.iloc[j]['STOI'])

            ratio_stoi_values.append(pw_ratios_stoi[i, j])

    #print("Costs: ", pw_ratios_stoi[row_ind, col_ind])
    #ipdb.set_trace()

    selected_pairs = pd.DataFrame({'low-stoi-para': low_stoi_paraphrases,
                                   'high-stoi-para': high_stoi_paraphrases,
                                   'low-stoi': low_stoi_values,
                                   'high-stoi': high_stoi_values,
                                   'ratio-stoi': ratio_stoi_values})

    selected_pairs['noise_type'] = gdf['noise_type'].iloc[0]
    selected_pairs['noise_level'] = gdf['noise_level'].iloc[0]

    return selected_pairs


def load_data(dfile,
              delim="\t",
              header_row=None,
              paraphrases_id_col='paraphrases_id',
              noise_types=None,
              noise_levels=None,
              pw_STOI_ratio_threshold=1.0,
              output_dir="/tmp"
              ):


    dataframe = pd.read_table(dfile,
                              sep=delim,
                              header=header_row)

    dataframe = dataframe.drop_duplicates()

    for each_col in dataframe.columns:
        for _ in range(2):
            dataframe[each_col] = dataframe[each_col].apply(lambda txt: txt.strip('"') if isinstance(txt, str) else txt)
            dataframe[each_col] = dataframe[each_col].apply(lambda txt: txt.strip("'") if isinstance(txt, str) else txt)

    #ipdb.set_trace()

    # 1.make pairs using a random noise-type and noise-level
    print("Start time of batch preparation :", time.ctime())

    # permutations
    total_no_pairs = sum([factorial(len(gdf))/factorial(len(gdf) - 2) for _, gdf in
                                            dataframe.groupby(paraphrases_id_col)])

    print("Total paraphrase pairs : ", total_no_pairs)

    new_dataframe = dataframe.groupby(paraphrases_id_col).apply(lambda gdf:
                                                              add_noise_cols(gdf,
                                                              group_id_col=paraphrases_id_col,
                                                              noise_levels=noise_levels,
                                                              noise_types=noise_types,
                                                              pw_STOI_ratio_threshold=pw_STOI_ratio_threshold,
                                                              expand=False))

    print("End time of batch preparation :", time.ctime())
    new_dataframe = new_dataframe.reset_index(drop=True)

    grouping_cols = ['noise_level', 'noise_type']  # ['noise_level']
    for gid, gdf in new_dataframe.groupby(grouping_cols):
        print("Group id: ", gid)
        print("Count: ", len(gdf), "~", 100 * (len(gdf)/total_no_pairs), "%")
        print(gdf[['low-stoi', 'high-stoi', 'ratio-stoi']].mean())

    # ipdb.set_trace()
    #make a a groupt histogram
    plot_file = os.path.join(output_dir, "histogram-pw-ratio.jpg")
    pG.plot_histogram(new_dataframe['ratio-stoi'],
                      xlabel="pair-wise STOI",
                      ylabel="Number of pairs",
                      title="A histogram of pair wise STOI ratio",
                      fileName=plot_file)


    new_dataframe.to_csv(os.path.join(output_dir, "dataset.tsv"), sep="\t", index=False)

    dataset = Dataset.from_pandas(new_dataframe)

    return dataset


#dfile ="/nethome/achingacham/PycharmProjects/LLaMA/llama-data/SWDA-PiSPIN/train.tsv"
#dfile = "/projects/SFB_A4/llama-2/llama-data/SWDA-PiSPIN/input_text_aa.txt-08-31-2023,202822_split_cols-text2speech_all.txt"
dfile = "/projects/SFB_A4/llama-2/llama-data/SWDA-PiSPIN/input_text_ALL.txt-09-17-2023,014627.tsv"

noise_dir = "/projects/SFB_A4/AudioRepo/noise_1/"
list_noise_files = [os.path.join(noise_dir, nFile) for nFile in os.listdir(noise_dir)]

#output_dir = "./results"
output_dir = "/projects/SFB_A4/llama-2/llama-ft-models/" + datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
os.mkdir(output_dir)

config_file = os.path.join(output_dir, "configs")
cFile = open(config_file, "w")
#cFile.write("\n dfile: " + dfile)


dataset = load_data(dfile=dfile,
          delim="\t",
          header_row=0,
          noise_types=list_noise_files,
          noise_levels=[-5],
          pw_STOI_ratio_threshold=1.0, output_dir=output_dir)


# dataset = dataset.rename_column('input_text', 'text')
# dataset = dataset.rename_column('best_hypo', 'label')

dataset = dataset.rename_column('low-stoi-para', 'input_text')
dataset = dataset.rename_column('high-stoi-para', 'best_hypo')

base_model_name = "meta-llama/Llama-2-7b-chat-hf"

#ipdb.set_trace()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = {"": 0}

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True
)

base_model.config.use_cache = False

# More info: https://github.com/huggingface/transformers/pull/24906
base_model.config.pretraining_tp = 1

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# def print_tokens_with_ids(txt):
#     tokens = tokenizer.tokenize(txt, add_special_tokens=False)
#     token_ids = tokenizer.encode(txt, add_special_tokens=False)
#     print(list(zip(tokens, token_ids)))
#
# prompt = \"""### User: Hello\n\n### Assistant: Hi, how can I help you?\"""
# print_tokens_with_ids(prompt)  # [..., ('▁Hello', 15043), ('<0x0A>', 13), ('<0x0A>', 13), ('##', 2277), ('#', 29937), ('▁Ass', 4007), ('istant', 22137), (':', 29901), ...]
# response_template = "### Assistant:"
# print_tokens_with_ids(response_template)
# response_template_with_context = "\n### Assistant:"
# print_tokens_with_ids(response_template_with_context)

def formatting_prompts_func(example):
    output_texts = []
    #p_c
    #instruction = "For a noisy listening environment with babble noise at SNR -5, generate a simple, intelligible, and spoken-styled paraphrase with 10-12 words, for the following input sentence: "
    #p_a
    instruction = "For the given input sentence, generate an intelligible paraphrase for a noisy environment with babble noise at snr -5:"

    for i in range(len(example['input_text'])):
        text = f"### Human: {instruction} {example['input_text'][i]}\n### Assistant: {example['best_hypo'][i]}"
        output_texts.append(text)
        cFile.write("\n" + text)

    return output_texts

instruction_template = "### Human:"
response_template_with_context = "\n### Assistant:"
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
# Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids,
                                           tokenizer=tokenizer)

#collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template_with_context, tokenizer=tokenizer, mlm=False)



training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8, #4
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=500,
    save_steps=200
)

max_seq_length = 512

#ipdb.set_trace()

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    #dataset_text_field="text",
)

trainer.train() #import inspect; inspect.getfile(trainer.train)

import os
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)



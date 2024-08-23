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
import argparse
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from math import factorial
from itertools import combinations, permutations
from datasets import load_dataset, Dataset, DatasetDict

from sfba4.utils import plotGraphs as pG
from sfba4.utils import create_mixed_audio_file as mix_audio
from sent_comp import get_features

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from transformers import EarlyStoppingCallback, IntervalStrategy
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



    #default arguments
    ma_args = {
        'clean_file': '/data/users/achingacham/Corpora/quora-question-pairs/qqp_audio_test/clean_audio/utt_6997.wav.wav',
        'noise_file': '/projects/SFB_A4/AudioRepo/noises_2/babble', 'output_mixed_file': '/tmp/sample_noisy_speech.wav',
        'output_clean_file': '/tmp/sample_clean.wav', 'snr': -5.0, 'startTime': 0, 'endTime': 100,
        'max_mix_length': None, 'stdOut': False, 'mixNoise': True}

    if clean_file is not None:
        ma_args['clean_file'] = clean_file

    if noise_type_file  is not None:
        ma_args['noise_file'] = noise_type_file

    if noise_level is not None:
        ma_args['snr'] = noise_level

    if noise_type_file_start is not None:
        ma_args['max_mix_length'] = noise_type_file_start

    # print(ma_args)
    # ipdb.set_trace()

    mix_audio.perform_additive_noise_mixing(**ma_args)

    stoi = get_features.get_STOI(ma_args['clean_file'], ma_args['output_mixed_file'])

    return stoi

def print_tokens_with_ids(txt):
    tokens = tokenizer.tokenize(txt, add_special_tokens=False)
    token_ids = tokenizer.encode(txt, add_special_tokens=False)
    print(list(zip(tokens, token_ids)))

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
              output_dir="/tmp",
              split="train"
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


    new_dataframe.to_csv(os.path.join(output_dir, split + "_dataset.tsv"), sep="\t", index=False)
    print("Saved dataset at: ", os.path.join(output_dir, split + "_dataset.tsv"))
    #dataset = Dataset.from_pandas(new_dataframe)
    #return dataset

    return new_dataframe

def formatting_prompts_func(example):
    output_texts = []
    # p_c
    # instruction = "For a noisy listening environment with babble noise at SNR -5, generate a simple, intelligible, and spoken-styled paraphrase with 10-12 words, for the following input sentence: "
    # p_a
    # instruction = "For the given input sentence, generate an intelligible paraphrase for a noisy environment with babble noise at snr -5:"

    for i in range(len(example['input_text'])):
        text = f"{example['promptPrefix'][i]}{example['input_text'][i]}{example['promptInfix'][i]}{example['best_hypo'][i]}"
        output_texts.append(text)
        lFile.write("\n" + text)

    return output_texts


if __name__ == '__main__':

    parser = argparse.ArgumentParser("a script to do supervised fine-tuning with a pseudo parallel dataset")

    parser.add_argument("-dFile", '--dataFile', help="path to the input data file(sentence mapped to clean utterance)",
                        default="/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/train_SWDA_00-05-text2speech_all.tsv"
                        # SWDA 00-03: "/projects/SFB_A4/llama-2/llama-data/SWDA-PiSPIN/input_text_ALL.txt-09-17-2023,014627.tsv"
                        )
    parser.add_argument("-sep", '--separator', help="delimiter for the data file(s)",
                        default="\t")
    parser.add_argument("-hRow", '--headerRow', help="row index for the header in data file (0 for the first line)",
                        type=int, default=0)

    parser.add_argument("-dsFile", '--dataSetFile', help="path to the input dataset file(with (pseudo) parallel data)",
                        default=None
                        # babble, SNR -5: "/projects/SFB_A4/llama-2/llama-ft-models/09-18-2023_03\:50\:51_SWDA_dataset/dataset.tsv"
                        # white, SNR -5: "/projects/SFB_A4/llama-2/llama-ft-models/09-23-2023_18:36:44_SWDA_dataset/all_dataset.tsv"
                        )
    parser.add_argument("-dsFile_iCol", '--dataSetFile_inputColumn', help="the input column name in the given dataset file",
                        default="low-stoi-para"
                        )
    parser.add_argument("-dsFile_oCol", '--dataSetFile_outputColumn', help="the input column name in the given dataset file",
                        default="high-stoi-para"
                        )

    parser.add_argument("-oDir", "--outputDirectory", help="path to a directory to saved fine-tuned models",
                        default="/projects/SFB_A4/llama-2/llama-ft-models/")

    parser.add_argument("-nDir", "--noiseDirectory", help="path to a directory with audio noise files",
                        default="/projects/SFB_A4/AudioRepo/noise_1/")
    parser.add_argument("-nLevels", "--noiseLevels", help="a comma separated list of SNR values",
                        default="-5")
    parser.add_argument("-tValue", "--thresholdValue", help="a float value",
                        default=1.0, type=float)
    parser.add_argument("-bModel", '--baseModel', help="the base model that exists in the HF hub",
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("-bSize", '--batchSize', help="train batch size",
                        default=8, type=int)
    parser.add_argument("-pPrefix", '--promptPrefix', help="mention the user prompt required for the API call",
                        default="### Human: For the given input sentence, generate an intelligible paraphrase for a noisy environment with babble noise at snr -5: "
                        )
    parser.add_argument("-pInfix", "--promptInfix", help="a keyword to identify the beginning of model output",
                        default="\n### Assistant: ")


    # parser.add_argument("-iCol", '--inputColumn', help="name/index of input column in data file",
    #                     default="input_text")
    # parser.add_argument("-oTCol", '--outputTextColumn', help="output column for text in the output file",
    #                     default="input_text")
    # parser.add_argument("-oPCol", '--outputParaColumn', help="output column for paraphrases in the output file",
    #                     default="system_response")
    # parser.add_argument("-oPidCol", '--outputParaIdColumn', help="output column for paraphrases id in the output file",
    #                     default="paraphrases_id")
    # parser.add_argument("-oFile", '--outputFile', help="path to the output file", default=None)


    args = parser.parse_args()
    


    dfile = args.dataFile
    #dfile ="/nethome/achingacham/PycharmProjects/LLaMA/llama-data/SWDA-PiSPIN/train.tsv"
    #dfile = "/projects/SFB_A4/llama-2/llama-data/SWDA-PiSPIN/input_text_aa.txt-08-31-2023,202822_split_cols-text2speech_all.txt"

    noise_dir = args.noiseDirectory
    list_noise_files = [os.path.join(noise_dir, nFile) for nFile in os.listdir(noise_dir)]

    output_dir = args.outputDirectory + datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    os.mkdir(output_dir)

    log_file = os.path.join(output_dir, "logs")
    lFile = open(log_file, "w")

    with open(os.path.join(output_dir, "args.log"), "w") as lF:
        for key, value in vars(args).items():
            lF.write("\n" + key + "\t:\t" + str(value))

    if args.dataSetFile is None:
        dataframe = load_data(dfile=dfile,
                  delim=args.separator,
                  header_row=args.headerRow,
                  noise_types=list_noise_files,
                  noise_levels=[float(s) for s in args.noiseLevels.split(',')],
                  pw_STOI_ratio_threshold=args.thresholdValue,
                  output_dir=output_dir,
                  split="all")

    else:
        dataframe = pd.read_table(args.dataSetFile)


    # ipdb.set_trace()
    # limit the samples with a threshold
    min_ratio_stoi = args.thresholdValue + 0.1

    if 'ratio-stoi' in dataframe.columns:
        dataframe = dataframe[dataframe['ratio-stoi'] >= min_ratio_stoi]
        print("Dataframe is filtered! With min_ratio_stoi of ", min_ratio_stoi,
              "the DF size is ", len(dataframe))

    # split 80-20 for train & dev
    train_dataframe = dataframe.sample(frac=0.8, random_state=42)
    test_dataframe = dataframe.drop(train_dataframe.index)

    ### add a stop sequence at the end of output text
    stop_sequence = "\n######"
    train_dataframe[args.dataSetFile_outputColumn] = train_dataframe[args.dataSetFile_outputColumn].apply(
        lambda txt: txt + stop_sequence)
    test_dataframe[args.dataSetFile_outputColumn] = test_dataframe[args.dataSetFile_outputColumn].apply(
        lambda txt: txt + stop_sequence)


    train_dataset = Dataset.from_pandas(train_dataframe)
    test_dataset = Dataset.from_pandas(test_dataframe)

    print("Size of train & test: ", len(train_dataframe), len(test_dataframe))
    #ipdb.set_trace()

    dataset_train_test = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    for ds_key, dataset in dataset_train_test.items():
        # rename columns, if required
        dataset = dataset.rename_column(args.dataSetFile_inputColumn,
                                        'input_text')
        dataset = dataset.rename_column(args.dataSetFile_outputColumn,
                                        'best_hypo')

        dataset = dataset.add_column('promptPrefix',
                                     [args.promptPrefix] * len(dataset))
        dataset = dataset.add_column('promptInfix',
                                     [args.promptInfix] * len(dataset))

        dataset.save_to_disk(os.path.join(output_dir, "model_dataset_" + ds_key))
        print("Dataset saved at: ", os.path.join(output_dir, "model_dataset_" + ds_key))


        dataset_train_test[ds_key] = dataset


    base_model_name = args.baseModel

    # model quantization for better resource management
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    device_map = {"": 0}

    # instantiate the base model
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

    #PEFT configurations
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none", #"lora_only", "all"
        task_type="CAUSAL_LM",
    )

    #setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token


    # prepare the data collator
    # instruction_template = "### Human:"
    response_template_with_context = args.promptInfix
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    #Now we have the response template tokens like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

    check_sent=args.promptPrefix+" a sample input sentence"+args.promptInfix +"i believe it was early in the summer"

    print_tokens_with_ids(check_sent)
    print_tokens_with_ids(response_template_with_context)

    #ipdb.set_trace()
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids,
                                               tokenizer=tokenizer)

    # for conversational-style collator
    #collator = DataCollatorForCompletionOnlyLM(
                                    # instruction_template=instruction_template,
                                    # response_template=response_template_with_context,
                                    # tokenizer=tokenizer,
                                    # mlm=False)


    batch_size = args.batchSize
    n_epochs = 10

    n_training_samples = dataset_train_test["train"].num_rows
    n_steps_per_epoch = int(n_training_samples/batch_size)
    n_max_steps = n_epochs * n_steps_per_epoch
    n_eval_steps = int(n_max_steps * 0.01)
    n_save_steps = 1 * n_eval_steps   # save_steps needs to be a whole multiple of eval_steps

    print("Total steps: ", n_max_steps,
          " | Eval steps: ", n_eval_steps,
          " | Save steps: ", n_save_steps)


    #setup training hyper-parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size, #or 4?
        gradient_accumulation_steps=4,
        learning_rate=1e-5, #2e-4,
        logging_steps=10,
        max_steps=n_max_steps, #500,
        save_steps=n_save_steps, #200
        save_strategy="steps",
        eval_steps=n_eval_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=5,
    )

    max_seq_length = 512


    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset_train_test["train"],
        eval_dataset=dataset_train_test["test"],
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        #dataset_text_field="text", # do not use this with data collator
    )

    #ipdb.set_trace()

    trainer.train() #import inspect; inspect.getfile(trainer.train)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

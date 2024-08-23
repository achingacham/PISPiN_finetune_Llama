# base code credits: https://huggingface.co/docs/trl/sft_trainer

import os
import re
import csv
import ipdb
import copy
import json
import torch
import time
import datetime

import argparse
import pandas as pd

from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers

from peft import AutoPeftModelForCausalLM

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class finetuned_model:

    def __init__(self, model_dir=None,
                 tokenizer=None,
                 disable_adapters=False,
                 ):

        self.disable_adapters = disable_adapters

        if model_dir is not None:

            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.model = AutoPeftModelForCausalLM.from_pretrained(model_dir,
                                                         device_map="auto",
                                                         torch_dtype=torch.float16,
                                                         #quantization_config=self.bnb_config
                                                         #load_in_4bit=True
                                                         )

            self.model.cuda()
            print_trainable_parameters(self.model)


            if tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)


    def get_model_response(self, text=None,
                           input_text="...",
                           promptPrefix='',
                           promptInfix="\n### Assistant: "
                           ):


        if text is None:
            #text = f"You are a helpful assistant.  {promptPrefix}{input_text}{promptInfix}"
            text = f"{promptPrefix}{input_text}{promptInfix}"
            ntokens_input_text = len(self.tokenizer(input_text)['input_ids'])
            min_ntokens_output = int(ntokens_input_text - (ntokens_input_text * 0.5))
            max_ntokens_output = int(ntokens_input_text + (ntokens_input_text * 0.5))

        else:
            ipdb.set_trace()
            ntokens_input_text = len(self.tokenizer(input_text)['input_ids'])
            min_ntokens_output = int(ntokens_input_text)
            max_ntokens_output = int(ntokens_input_text)

        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        if self.disable_adapters:
            self.model.disable_adapter_layers()

        outputs = self.model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                      attention_mask=inputs["attention_mask"],
                                      max_new_tokens=max_ntokens_output,
                                      min_new_tokens=min_ntokens_output,
                                      pad_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)


        #ipdb.set_trace()
        #return response.split(promptInfix)[1]
        #print(input_text, "->", response.split(promptInfix))
        #return response.split(promptInfix)[1].split('###')[0].strip('\n').strip()
        #return response.split(promptInfix)[1].split('\n')[0].split('###')[0].strip()
        return response.split(promptInfix)[1].split('\n######')[0].strip() #use the same stop sequence used in training.

if __name__ == '__main__':

    parser = argparse.ArgumentParser("a script to do inference from a fie-tned LLaMA model")

    parser.add_argument("-iFile", '--inputFile', help="path to the input data file",
                        default="/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv"
                        )
    parser.add_argument("-sep", '--separator', help="delimiter for the data file(s)",
                        default="\t")
    parser.add_argument("-hRow", '--headerRow', help="row index for the header in  data file (0 for the first line)",
                        type=int, default=0)
    parser.add_argument("-iCol", '--inputColumn', help="name/index of input column in data file",
                        default="input_text")
    parser.add_argument("-pPrefix", '--promptPrefix', help="mention the user prompt required for the API call",
                        default="For the given input sentence, generate an intelligible paraphrase for a noisy environment with babble noise at snr -5: "
                        )
    parser.add_argument("-pInfix", "--promptInfix", help="a keyword to identify the beginning of model output",
                        default="\n### Assistant: ")

    parser.add_argument("-mPath", '--modelPath', help="path to the fine-tuned model for paraphrase generation",
                        default="/projects/SFB_A4/llama-2/llama-ft-models/09-17-2023_00:29:00")

    parser.add_argument('-disableAL', '--disableAdapterLayers',
                        help="a flag to include/exclude adapter layers",
                        action="store_true")

    parser.add_argument("-oTCol", '--outputTextColumn', help="output column for text in the output file", default="input_text")
    parser.add_argument("-oPCol", '--outputParaColumn', help="output column for paraphrases in the output file", default="system_response")
    parser.add_argument("-oPidCol", '--outputParaIdColumn', help="output column for paraphrases id in the output file", default="paraphrases_id")

    parser.add_argument("-oFile", '--outputFile', help="path to the output file", default=None)

    args = parser.parse_args()


    # set some variables before script execution:
    exec_ts = datetime.datetime.now()

    if args.outputFile is None:
        outputFile = args.inputFile + "_ft_model_" + "-".join(args.modelPath.split('/')[-2:]) + "_" + exec_ts.strftime("%m-%d-%Y,%H%M%S")
    else:
        outputFile = args.outputFile

    print("Prompt prefix and infix: ", args.promptPrefix, "\n", args.promptInfix)
    print("Check out the output file: ", outputFile)

    logFile = outputFile + ".json"
    if os.path.exists(logFile):
        os.remove(logFile)

    if args.inputFile is not None:
        input_dataset = pd.read_table(args.inputFile, sep=args.separator, header=args.headerRow) #quoting=csv.QUOTE_NONE
    else:
        print("Check the path to the input file.")
        raise NotImplementedError

    if args.promptPrefix is None:
        print("Check the user prompt. its not given!")
        raise NotImplementedError

    outputTextColumn = args.outputTextColumn
    outputParaColumn = args.outputParaColumn
    outputParaIdColumn = args.outputParaIdColumn

    if args.headerRow is None:
        output_dataset = pd.DataFrame(input_dataset[int(args.inputColumn)-1].tolist(), columns=[outputTextColumn]) #when header is None, columns are referred using indices
    else:
        output_dataset = pd.DataFrame(input_dataset[args.inputColumn].tolist(), columns=[outputTextColumn]) # else, just refer with the column name.

    if outputParaIdColumn not in input_dataset.columns:
        output_dataset[outputParaIdColumn] = output_dataset.index
    else:
        output_dataset[outputParaIdColumn] = input_dataset[outputParaIdColumn]

    #ipdb.set_trace()

    #remove additional spaces in textColumn
    output_dataset[outputTextColumn] = output_dataset[outputTextColumn].apply(lambda txt: re.sub(r' {2,}',' ', txt))

    # generate the user prompt content:
    output_dataset['user_prompt'] = output_dataset[outputTextColumn].apply(
        lambda txt: args.promptPrefix + txt + args.promptInfix)

    #ipdb.set_trace()
    ft_model = finetuned_model(model_dir=args.modelPath, disable_adapters=args.disableAdapterLayers)

    output_dataset[outputParaColumn] = output_dataset[outputTextColumn].apply(lambda iText:
                                                                              ft_model.get_model_response(
                                                                            input_text=iText,
                                                                            promptPrefix=args.promptPrefix,
                                                                            promptInfix=args.promptInfix))


    #testing on output length:

    output_dataset['output_length'] = output_dataset[outputParaColumn].apply(lambda txt: len(txt))
    print("### Generated paraphrase with maximum and minimum length. ###")
    print(output_dataset[output_dataset['output_length'] == output_dataset['output_length'].max()]
          [outputParaColumn].iloc[0])
    print(output_dataset[output_dataset['output_length'] == output_dataset['output_length'].min()]
          [outputParaColumn].iloc[0])

    # ipdb.set_trace()
    exec_te = datetime.datetime.now()

    print("Total time taken (ms): ", exec_te-exec_ts)
    output_dataset['user_prompt'] = output_dataset['user_prompt'].apply(lambda txt: re.sub(r'\n', r'\\n', txt))
    output_dataset[outputParaColumn] = output_dataset[outputParaColumn].apply(lambda x: x.replace('"', ''))
    output_dataset[outputParaColumn] = output_dataset[outputParaColumn].apply(lambda txt: re.sub(r'\n', '', txt))
    output_dataset.to_csv(outputFile, sep="\t", index=False)

    print("Check out the output file: ", outputFile)

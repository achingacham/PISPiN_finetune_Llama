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

from transformers import AutoTokenizer
import transformers

#print("Start time: ", time.ctime())


class llamaGeneration:

    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", prompt_infix="\n\Answer:\n"):
        self.model = model
        self.prompt_infix = prompt_infix
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.pipeline = transformers.pipeline(
                        "text-generation",
                        model=model,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        )

        #print("Model loaded at: ",  time.ctime())
        #ipdb.set_trace()

    def generate(self, prompt, logFile="/tmp/llama_generation.log"):

        # print(prompt)
        # ipdb.set_trace()

        chat_response = self.pipeline(
        #'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
        prompt,
        do_sample=False, #set this to False for deterministic output
        top_k=10,
        num_return_sequences=1,
        eos_token_id=self.tokenizer.eos_token_id,
        max_length=300,
        )



        # print(chat_response)
        # ipdb.set_trace()
        # for seq in chat_response:
        #     #print(seq['generated_text'])

        lFile = open(logFile, "a")
        list_responses = []
        for seq in chat_response:
            json.dump(seq, lFile)
            #split_gen_text = seq['generated_text'].split(self.prompt_infix.strip('"'))

            # if len(split_gen_text) != 2:
            #     ipdb.set_trace()
            #     print(split_gen_text)
            #     response = seq['generated_text']
            # else:
            #     response = split_gen_text[1]

            response = seq['generated_text'].split(self.prompt_infix)[1].split('\n')[0]
            list_responses.append(response.strip())
            lFile.write("\n")

        lFile.close()

        # ipdb.set_trace()
        return list_responses[0]  # only a single response seq is expected as of now.

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-iFile", '--inputFile', help="path to the input data file",
                        default="/projects/SFB_A4/A4-IntelligibleParaphrasesinNoise/data/HE_300_600/SWDA_short_utterances_300_600.tsv"
                        )
    parser.add_argument("-sep", '--separator', help="delimiter for the data file(s)",
                        default="\t")
    parser.add_argument("-hRow", '--headerRow', help="row index for the header in  data file (0 for the first line)",
                        type=int, default=0)
    parser.add_argument("-iCol", '--inputColumn', help="name/index of input column in data file",
                        default="input_text")
    # parser.add_argument("-uPrompt", '--userPrompt', help="mention the user prompt required for the API call",
    #                     default=None#"Paraphrase the following sentence: "
    #                     )
    # parser.add_argument("--kw_split_prompt_response", help="a keyword to identify the beginning of model output",
    #                     default="\n\nAnswer:\n")

    parser.add_argument("-pPrefix", '--promptPrefix', help="mention the user prompt required for the API call",
                        default="For the given input sentence, generate an intelligible paraphrase for a noisy environment with babble noise at snr -5: "
                        )
    parser.add_argument("-pInfix", "--promptInfix", help="a keyword to identify the beginning of model output",
                        default="\n### Assistant: ")

    parser.add_argument("-mName", '--modelName', help="name the model for paraphrase generation",
                        default="meta-llama/Llama-2-7b-chat-hf")

    parser.add_argument("-oTCol", '--outputTextColumn', help="output column for text in the output file", default="input_text")
    parser.add_argument("-oPCol", '--outputParaColumn', help="output column for paraphrases in the output file", default="system_response")
    parser.add_argument("-oPidCol", '--outputParaIdColumn', help="output column for paraphrases id in the output file", default="paraphrases_id")

    parser.add_argument("-oFile", '--outputFile', help="path to the output file", default=None)

    args = parser.parse_args()

    # set some variables before script execution:
    exec_ts = datetime.datetime.now()

    if args.outputFile is None:
        outputFile = args.inputFile + "-" + exec_ts.strftime("%m-%d-%Y,%H%M%S")
    else:
        outputFile = args.outputFile

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
    #ipdb.set_trace()

    # generate the user prompt content:
    # kw_split_prompt_response = args.kw_split_prompt_response
    output_dataset['user_prompt'] = output_dataset[outputTextColumn].apply(
        lambda txt: args.promptPrefix + txt + args.promptInfix)

    llm_g = llamaGeneration(model=args.modelName, prompt_infix=args.promptInfix)
    output_dataset[outputParaColumn] = output_dataset['user_prompt'].apply(lambda uprompt: llm_g.generate(uprompt, logFile))

    #testing on output length:
    output_dataset['output_length'] = output_dataset[outputParaColumn].apply(lambda txt: len(txt))
    print(output_dataset[output_dataset['output_length'] == output_dataset['output_length'].max()][
              outputParaColumn].iloc[0])

    # ipdb.set_trace()
    exec_te = datetime.datetime.now()

    print("Total time taken (ms): ", exec_te-exec_ts)
    output_dataset['user_prompt'] = output_dataset['user_prompt'].apply(lambda txt: re.sub(r'\n', r'\\n', txt))
    output_dataset[outputParaColumn] = output_dataset[outputParaColumn].apply(lambda x: x.replace('"', ''))
    output_dataset[outputParaColumn] = output_dataset[outputParaColumn].apply(lambda txt: re.sub(r'\n', r'\\n', txt))
    output_dataset.to_csv(outputFile, sep="\t", index=False)

    print("Output file name starts with: " + outputFile)
    ipdb.set_trace()

    # when multiple responses are requested, a list in returned with '\n' delimiter is the output
    new_output_dataset_diffrow = []
    new_output_dataset_diffcol = []

    n_sys_responses = []
    line_delimiter = ""  # args.outputParaColumnDelim

    if "\\n" in output_dataset[outputParaColumn].iloc[0]:
        line_delimiter = "\\n"
        another_line_delimiter = "\n"
    if "\n" in output_dataset[outputParaColumn].iloc[0]:
        line_delimiter = "\n"
        another_line_delimiter = "\\n"

    if line_delimiter != "":
        for row_idx, each_row in output_dataset.iterrows():
            input_text = each_row[outputTextColumn]
            user_prompt = each_row['user_prompt']

            # if re.findall(" That just shows you how much I've been paying attention", user_prompt):
            #     ipdb.set_trace()

            # systems_responses = [each_resp.split(' ', 1)[1] for each_resp in
            #                      re.sub('\n+', '\n', each_row[outputParaColumn]).split('\n') if
            #                      len(each_resp.split(' ', 1)) == 2]  # the first split is the index

            systems_responses = [re.split(r'^{}*\s*\d\.\s'.format(line_delimiter), each_resp)[1].strip() for each_resp
                                 in
                                 re.sub('{}+'.format(line_delimiter), line_delimiter, each_row[outputParaColumn]).
                                     split(line_delimiter) if len(
                    re.split(r'^{}*\s*\d\.\s'.format(line_delimiter), each_resp)) == 2]  # the first split is the index

            if len(systems_responses) == 1:
                # print(systems_responses)
                systems_responses = [re.split(r'^{}*\s*\d\.\s'.format(another_line_delimiter), each_resp)[1].strip() for
                                     each_resp in re.sub('{}+'.format(another_line_delimiter), another_line_delimiter,
                                                         each_row[outputParaColumn]).
                                         split(another_line_delimiter) if
                                     len(re.split(r'^{}*\s*\d\.\s'.format(another_line_delimiter),
                                                  each_resp)) == 2]  # the first split is the index

                # print(systems_responses)

            if row_idx == 0:
                expected_num_resp = len(systems_responses)

            n_sys_responses.append(len(systems_responses))

            # if len(systems_responses) != 6:
            #     ipdb.set_trace()

            list_sys_responses = ["-" for i in range(expected_num_resp)]

            for sys_resp_idx, each_sys_resp in enumerate(systems_responses):

                if each_sys_resp != '':
                    new_output_dataset_diffrow.append([row_idx, input_text, user_prompt, each_sys_resp])

                # if more than 'expected_num_resp', ignore
                if sys_resp_idx < expected_num_resp:
                    list_sys_responses[sys_resp_idx] = each_sys_resp

            new_output_dataset_diffcol.append([row_idx, input_text, user_prompt] + list_sys_responses)

        output_dataset = pd.DataFrame(new_output_dataset_diffrow,
                                      columns=[outputParaIdColumn, outputTextColumn, 'user_prompt', outputParaColumn])

        output_dataset[outputParaColumn] = output_dataset[outputParaColumn].apply(lambda x: x.replace('"', ''))
        output_dataset.to_csv(outputFile + "_split_rows", sep="\t",
                              index=False)

        if len(set(n_sys_responses)) != 1:
            from collections import Counter

            print("Different number of output responses! : ", Counter(n_sys_responses))
            # ipdb.set_trace()

        output_dataset = pd.DataFrame(new_output_dataset_diffcol,
                                      columns=[outputParaIdColumn, outputTextColumn, 'user_prompt'] + [
                                          outputParaColumn + '_' + str(i + 1) for i in range(n_sys_responses[0])])

        for each_col in [outputParaColumn + '_' + str(i + 1) for i in range(n_sys_responses[0])]:
            output_dataset[each_col] = output_dataset[each_col].apply(lambda x: x.replace('"', ''))

        output_dataset.to_csv(outputFile + "_split_cols", sep="\t",
                              index=False)

        print("Output file name starts with: " + outputFile)


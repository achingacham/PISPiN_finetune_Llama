import os
import ipdb
import datetime
import argparse
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser("a script to combine N files (of same structure) with a modified unique identifier")

    parser.add_argument("--inputFilesDirectory", "-iFilesDir",
                        default=None,
                        help="if not None, use this with 'inputFiles' list for absolute path")
    parser.add_argument("--inputFiles", "-iFiles",
                        help="a semi-colon (;) separated list of input files (absolute/relative path)")
    parser.add_argument("--inputFilesDelimiter", "-iFilesDelim",
                        help="input files delimiter", default="\t")
    parser.add_argument("-hRow", '--headerRow', help="row index for the header in  data file (0 for the first line)",
                        type=int, default=0)
    parser.add_argument("--inputFilesUniqueIdCol", "-iFilesUidCol",
                        help="the unique id (integer) column to be modified",
                        default="paraphrases_id")
    parser.add_argument("--outputFile", "-oFile",
                        help="the output file path to save the combined format",
                        default=None)

    args = parser.parse_args()
    exec_ts = datetime.datetime.now()

    #print(args)

    if args.outputFile is None:
        outputFile = os.path.join(args.inputFilesDirectory, "combined_data_" + exec_ts.strftime("%m-%d-%Y,%H%M%S") + ".tsv")
    else:
        outputFile = args.outputFile

    if args.inputFiles is None:
        print("Check the path to the input file.")
        raise NotImplementedError
    else:
        if args.inputFilesDirectory is None:
            list_input_files = args.inputFiles.split(';')
        else:
            list_input_files = [os.path.join(args.inputFilesDirectory, iFile) for iFile in args.inputFiles.split(';')]

        list_dataframes = []
        max_id = 0

        for fid, input_file in enumerate(list_input_files):

            input_dataset = pd.read_table(input_file,
                                        sep=args.inputFilesDelimiter,
                                        header=args.headerRow)  # quoting=csv.QUOTE_NONE


            print("Size of input dataset (with duplicates): ", len(input_dataset))
            input_dataset = input_dataset.drop_duplicates()

            print("Size of input dataset (without duplicates): ", len(input_dataset))

            input_dataset['file_id'] = fid
            # ipdb.set_trace()

            input_dataset[args.inputFilesUniqueIdCol] = input_dataset[args.inputFilesUniqueIdCol].apply(lambda x: x + max_id)
            list_dataframes.append(input_dataset)
            max_id = input_dataset[args.inputFilesUniqueIdCol].max() + 1

        output_dataset = pd.concat(list_dataframes)
        output_dataset.to_csv(outputFile, sep="\t", index=False)

        print("Total number of sets of paraphrases : ", len([_ for  _ in output_dataset.groupby('paraphrases_id')]))

        #ipdb.set_trace()

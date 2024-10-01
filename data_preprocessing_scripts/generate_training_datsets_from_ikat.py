import csv
import json
import os
from argparse import ArgumentParser

from tqdm import tqdm

INPUT_FILE = "/data/rech/huiyuche/ikat_official_collectioin/ikat_2023_passages_00.jsonl"
OUTPUT_FOLDER = "../test_text_dataset/train"
NUM_FILES = 1000


def main(input_file, output_file): 
    with open(INPUT_FILE, 'r') as input:
        for i in range(NUM_FILES):
            line = input.readline()
            dict_line = json.loads(line)
            file_name = f"{OUTPUT_FOLDER}/{dict_line['id']}.txt"
            file_content = dict_line["contents"]
            with open(file_name, 'w') as output:
                output.write(file_content)

if __name__ == "__main__":
    main(INPUT_FILE, OUTPUT_FOLDER)
import os
import time
import torch
import sys
from transformers import GPT2Config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import random
from scipy.stats import zscore

sys.path.append("./config") 
from model import * 
from data import *
from config_embedding_test import *

def read_bytes(filename, PATCH_SIZE, PATCH_LENGTH):
    
    ext = filename.split('.')[-1]
    ext = bytearray(ext, 'utf-8')
    ext = [byte for byte in ext][:PATCH_SIZE]

    with open(filename, 'rb') as f:
        file_bytes = f.read()

    bytes = []
    for byte in file_bytes:
        bytes.append(byte)

    # padding. For each patch, if do not have enough bytes, pad with 256.
    if len(bytes)%PATCH_SIZE!=0:
        bytes = bytes + [256] * (PATCH_SIZE - len(bytes) % PATCH_SIZE)

    bos_patch = ext + [256] * (PATCH_SIZE - len(ext))
    # 1. also add a bos patch at the beginning, as in inference.
    # 2. add an eos patch at the end to indicate the end of the sequence.
    bytes = bos_patch + bytes + [256] * PATCH_SIZE

    # TODO: the meaning of this truncation is not clear.
    # XXX Q1: shouldn't the training file already be split to the desired length?
    # XXX Q2: what if we truncate the head? it will lose the extension info.

    # "head": Keep the beginning part of the byte sequence.
    # "body": Keep a random middle segment of the byte sequence.
    # "tail": Keep the ending part of the byte sequence.

    if len(bytes) > PATCH_LENGTH*PATCH_SIZE:
        ## TODO: add the param here
        if False:
            warnings.warn(f"Warning: {filename} is too long, truncating to {PATCH_LENGTH*PATCH_SIZE} bytes.")
        choices = ["head", "body", "tail"]
        choice = random.choice(choices)
        if choice == "head":
            bytes = bytes[:PATCH_LENGTH*PATCH_SIZE]
        elif choice == "body" and len(bytes) > (PATCH_LENGTH+1)*PATCH_SIZE:
            start = random.randint(1, len(bytes)//PATCH_SIZE-PATCH_LENGTH)
            bytes = bytes[start*PATCH_SIZE:(start+PATCH_LENGTH)*PATCH_SIZE]
        else:
            bytes = bytes[-PATCH_LENGTH*PATCH_SIZE:]

    # TODO: what is the meaning of this mask? seems useless?
    masks = [1] * (len(bytes)//PATCH_SIZE)

    return bytes, masks

##########################################################
# Byte Frequency Distribution
##########################################################

class ByteFrequencyDistribution():
    """Get byte frequency distribution table & shannon entropy results."""

    def get_fingerprint_by_folder(self, folder_path):
        """Get byte frequency distribution table & correlation matrix.
        @param foler_path: folder path.
        @return: fingerprint dict or None.
        """

        file_list = os.listdir(folder_path)
        file_list = [file for file in file_list if os.path.isfile(os.path.join(folder_path, file))]
        list_of_byte_frequency_table = []
        for file in file_list:
            byte_frequency_table, _ = self.get_byte_frequency_table_by_file_path(os.path.join(folder_path, file))
            if byte_frequency_table:
                list_of_byte_frequency_table.append(byte_frequency_table)
        
        average_byte_frequency_table = {k : sum(d[k] for d in list_of_byte_frequency_table) / len(list_of_byte_frequency_table) for k in list_of_byte_frequency_table[0].keys()}

        # transfomrm list of dict to pandas dataframe
        list_of_str_byte_frequency_table = [{str(k): v for k, v in d.items()} for d in list_of_byte_frequency_table]
        df = pd.DataFrame(list_of_str_byte_frequency_table)
        # filter out columns with 0 frequency
        df_new = df.loc[:, (df != 0).any(axis=0)]
        # calculate correlation matrix
        corr = df.corr()
        corr_new = df_new.corr()
        return corr, corr_new, average_byte_frequency_table


    def get_BFD_Corr_plots(self, data_type, input_folder_path, output_folder_path, threshold=0.7):
        """Get byte frequency distribution table & correlation matrix.
        @param data_type: data type, str..
        @param input_folder_path: input folder path.
        @param output_folder_path: output folder path.
        @return: None.
        """

        corr, corr_new, average_byte_frequency_table = self.get_fingerprint_by_folder(input_folder_path)
        corr.fillna(-0.5, inplace=True)
        ##################################################
        ############# plot correlation matrix
        ##################################################

        title_full = f'Full Correlation Matrix of Datatype [{data_type}]'
        title_truncated = f'Core Correlation Matrix of Datatype [{data_type}]'

        output_path_full = os.path.join(output_folder_path, f'{data_type}_corr_full.png')
        output_path_truncated = os.path.join(output_folder_path, f'{data_type}_corr_core.png')

        self.draw_corr(corr, title_full, threshold, output_path_full)
        self.draw_corr(corr_new, title_truncated, threshold, output_path_truncated)

        #########################################################
        ############# plot Byte Frequency Distribution matrix
        #######################################################

        title = f'Byte Frequency Distribution of Datatype [{data_type}]'
        output_path = os.path.join(output_folder_path, f'{data_type}_average_BFD.png')
        self.draw_BFD(average_byte_frequency_table, title, output_path)

    def draw_BFD(self, byte_frequency_table, title, output_path):
        """Draw byte frequency distribution table.
        @param byte_frequency_table: byte frequency table dict.
        @param output_path: output path.
        @return: None.
        """

        # plot byte frequency distribution
        plt.figure(figsize=(15, 12))
        sns.set_style("white")
        df = pd.DataFrame(list(byte_frequency_table.items()), columns=['Byte Value', 'Frequency'])
        sns.barplot(
                x='Byte Value',              # x-axis represents byte values (from 00 to FF)
                y='Frequency',         # y-axis represents their corresponding frequency in percentage
                data=df,               # data source is our DataFrame created from the frequency table
                width=1,
                color='blue',           # color of the bars
            )

        # Define the specific x-ticks you want to show (e.g., 1, 32, 64, etc.)
        byte_values_too_big = [i for i in range(256) if byte_frequency_table[str(format(i,'02X'))] > 0.4]
        xticks = list(range(0,256,32)) + [255]

        final_xticks = sorted(list(set(byte_values_too_big + xticks)))
        

        # Set the x-ticks as well as y-ticks to only show those values
        plt.xticks(ticks=final_xticks)
        plt.yticks(ticks=list(np.arange(0, 1.01, 0.1)))

        plt.title(title)
        plt.savefig(output_path)
        plt.close()


    def draw_corr(self, corr, title, threshold, output_path):

        plt.figure(figsize=(15, 12))
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        # all values > 0.5 
        result = [(column, index) for column in upper.columns for index in upper.index if upper[column][index] > threshold]

        # variable names
        variable_names = set([name for pair in result for name in pair])

        #  get all ticks positions
        all_ticks = np.arange(len(corr.columns))

        #  only keep the ticks of variables with correlation > 0.5
        filtered_ticks = [i for i, col in enumerate(corr.columns) if col in variable_names]

        # Only display labels greater than 0.5, and set other labels to empty strings
        xticks_labels = ['' if i not in filtered_ticks else corr.columns[i] for i in all_ticks]
        yticks_labels = ['' if i not in filtered_ticks else corr.index[i] for i in all_ticks]

        # only show half of the matrix
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask = mask, annot=False, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        # Draw the heatmap with the mask and correct aspect ratio
        plt.title(title)
        plt.xticks(ticks=all_ticks, labels=xticks_labels, rotation=45)
        plt.yticks(ticks=all_ticks, labels=yticks_labels, rotation=45)

        plt.savefig(output_path)
        plt.close()

    def get_byte_frequency_table_by_byte_list(self, byte_stream):
        """Get byte frequency table dict.
        @param byte_stream: List[int], int value vaires from 0 to 255.
        @return: byte frequency table(over 0) dict or None.
        """

        byte_arr = byte_stream
        frequency_table = {}
        filesize = len(byte_arr)
        numerical_frequency_value = [0]*256
        for byte in byte_arr:
            numerical_frequency_value[byte] += 1
        max_frequency = max(numerical_frequency_value)
        numerical_frequency_value = [round(float(value),3) / max_frequency for value in numerical_frequency_value]
        numerical_frequency_table = {i: numerical_frequency_value[i] for i in range(256)}
        frequency_table = {str(format(i,'02X')): numerical_frequency_value[i] for i in range(256)}

        return frequency_table, numerical_frequency_table

    def get_byte_frequency_table_by_file_path(self, file_path):
        """Get byte frequency table dict.
        @param file_path: file path.
        @return: byte frequency table(over 0) dict or None.
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
        except IOError as e:
            print(e)
            return {}
        byte_arr = [byte for byte in data]

        frequency_table, numerical_frequency_table = self.get_byte_frequency_table_by_byte_list(byte_arr)
        return frequency_table, numerical_frequency_table

##########################################################
# Embedding Analysis
##########################################################

PATCH_NUM_LAYERS = 12
BYTE_NUM_LAYERS = 3
PATCH_LENGTH = 512
PATCH_SIZE = 16
PATCH_SAMPLING_BATCH_SIZE = 0
HIDDEN_SIZE = 768

def load_pretrained_patch_decoder(inference_weights_path):

    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    patch_config = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS, 
        max_length=PATCH_LENGTH, 
        max_position_embeddings=PATCH_LENGTH,
        hidden_size=HIDDEN_SIZE,
        n_head=HIDDEN_SIZE//64,
        vocab_size=1)   

    byte_config = GPT2Config(
        num_hidden_layers=BYTE_NUM_LAYERS, 
        max_length=PATCH_SIZE+1, 
        max_position_embeddings=PATCH_SIZE+1,
        hidden_size=HIDDEN_SIZE,
        n_head=HIDDEN_SIZE//64,
        vocab_size=256+1) # vocal size is all possible values of a byte plus 1 for eos.

    model = bGPTLMHeadModel(patch_config, byte_config, PATCH_SIZE, PATCH_SAMPLING_BATCH_SIZE, "none")
    print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    checkpoint = torch.load(inference_weights_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    patch_level_decoder = model.patch_level_decoder

    return patch_level_decoder

def inference_1_vector(byte_array, patch_model, PATCH_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    byte_array = torch.tensor([byte_array]).to(device)
    patch_array = byte_array.reshape(len(byte_array), -1, PATCH_SIZE)
    with torch.no_grad():
        output = patch_model(patch_array)
    
    last_hidden_state = output["last_hidden_state"]

    last_embedding = last_hidden_state[:, -1, :].reshape(1, -1).detach().cpu().numpy()
    return last_embedding


def get_embedding_value_distribution(data_type, input_folder_path, output_folder_path, handle_outlier="none"):
    """Get byte frequency distribution table & correlation matrix.
    @param data_type: data type, str..
    @param input_folder_path: input folder path.
    @param output_folder_path: output folder path.
    @ param handle_outlier: how to handle outliers, str.
    @return: None.
    """
    if data_type == "txt":
        inference_weights_path = "/data/rech/huiyuche/huggingface/bgpt/weights-text.pth"
    elif data_type == "png":
        inference_weights_path = "/data/rech/huiyuche/huggingface/bgpt/weights-image.pth"
    elif data_type == "flac":
        inference_weights_path = "/data/rech/huiyuche/huggingface/bgpt/weights-audio.pth"

    patch_level_decoder = load_pretrained_patch_decoder(inference_weights_path=inference_weights_path)

    all_numbers = np.array([[0]*HIDDEN_SIZE]) 

    for file in os.listdir(input_folder_path):
        # TODO: check the checkpoint is the pretrained one or the fine-tuned one.
        if os.path.isfile(os.path.join(input_folder_path, file)):
            bytes, _ = read_bytes(f"{input_folder_path}/{file}", PATCH_SIZE, PATCH_LENGTH)
            # trancate last patch of 256
            bytes = bytes[:-PATCH_SIZE]
            last_embedding = inference_1_vector(bytes, patch_level_decoder, PATCH_SIZE)
            assert last_embedding.shape == (1, HIDDEN_SIZE)
            last_embedding = outlier_handler(last_embedding, handle_outlier)
            all_numbers = np.concatenate((all_numbers, last_embedding), axis=0)

    all_numbers = all_numbers[1:]

    # 1. max, min, mean, std
    max_val = np.max(all_numbers)
    min_val = np.min(all_numbers)
    mean_val = np.mean(all_numbers)
    std_val = np.std(all_numbers)

    # print how many data points are there
    print(f"Number of data points: {all_numbers.size}")

    # max, min, mean, std
    print(f"Max: {max_val}, Min: {min_val}, Mean: {mean_val}, Std: {std_val}")

    # percentage of values in the range of [mean-std, mean+std]
    percentage = np.mean((all_numbers > mean_val - std_val) & (all_numbers < mean_val + std_val))
    print(f"Percentage of values in the range of [mean-std, mean+std]: {percentage}")

    # percentage of values in the range of [mean-2*std, mean+2*std]
    percentage = np.mean((all_numbers > mean_val - 2*std_val) & (all_numbers < mean_val + 2*std_val))
    print(f"Percentage of values in the range of [mean-2*std, mean+2*std]: {percentage}")

    # percentage of values in the range of [mean-3*std, mean+3*std]
    percentage = np.mean((all_numbers > mean_val - 3*std_val) & (all_numbers < mean_val + 3*std_val))
    print(f"Percentage of values in the range of [mean-3*std, mean+3*std]: {percentage}")

    # 2. Barplot
    flattened_embeddings = all_numbers.flatten()
    bin_counts, bin_edges = np.histogram(flattened_embeddings, bins=20)
    bin_centers = np.round((bin_edges[:-1] + bin_edges[1:]) / 2, 1)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=bin_centers, y=bin_counts)
    plt.title(f"Barplot of Embedding Values Distribution for {data_type}, handle_outlier={handle_outlier}")
    plt.xlabel("Embedding Values")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_folder_path}/{data_type}_barplot_{handle_outlier}.png")



    # 3. Distribution curve (KDE plot)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(flattened_embeddings, fill=True)
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    plt.title(f"Distribution Curve (KDE) of Embedding Values for {data_type}, handle_outlier={handle_outlier}")
    plt.xlabel("Embedding Values")
    plt.ylabel("Density")
    plt.savefig(f"{output_folder_path}/{data_type}_kdeplot_{handle_outlier}.png")

    # 4. Barplot for partial data -- mean-std, mean+std
    partial_embeddings = all_numbers[(all_numbers > mean_val - std_val) & (all_numbers < mean_val + std_val)]
    partial_flattened_embeddings = partial_embeddings.flatten()
    bin_counts, bin_edges = np.histogram(partial_flattened_embeddings, bins=20)
    bin_centers = np.round((bin_edges[:-1] + bin_edges[1:]) / 2, 1)


    plt.figure(figsize=(10, 6))
    sns.barplot(x=bin_centers, y=bin_counts)
    plt.title(f"Barplot of Embedding Values Distribution (mean-std, mean+std) for {data_type}, handle_outlier={handle_outlier}")
    plt.xlabel("Embedding Values")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_folder_path}/{data_type}_barplot_fine_grained_{handle_outlier}.png")

    # 5. Distribution curve (KDE plot) for partial data -- mean-std, mean+std
    plt.figure(figsize=(10, 6))
    sns.kdeplot(partial_flattened_embeddings, fill=True)
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    plt.title(f"Distribution Curve (KDE) of Embedding Values (mean-std, mean+std) for {data_type},handle_outlier={handle_outlier}")
    plt.xlabel("Embedding Values")
    plt.ylabel("Density")
    plt.savefig(f"{output_folder_path}/{data_type}_kdeplot_fine_grained_{handle_outlier}.png")

    return all_numbers

#############################################################
############### bits per byte evaluation ####################
#############################################################


def eval_epoch(eval_set, model):
    print(len(eval_set))
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 0
    model.eval()
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set: 
        input_patches, input_masks = batch
        input_patches = input_patches.to(model.device)
        input_masks = input_masks.to(model.device)
        with torch.no_grad():
            cross_entropy_loss = model(input_patches, input_masks).loss
        bits_per_bits_loss = cross_entropy_loss.item() / np.log(2)
        total_eval_loss += bits_per_bits_loss 
        iter_idx += 1

    return total_eval_loss / iter_idx


def bits_per_byte_evaluation():
    """
    calculate the bits per byte metrics given a config file.
    """
    print("the batch size is {}".format(BATCH_SIZE))

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    patch_config = GPT2Config(vocab_size=1,
                            n_positions=PATCH_LENGTH,
                            n_embd=HIDDEN_SIZE,
                            n_layer=PATCH_NUM_LAYERS,
                            n_head=HIDDEN_SIZE//64,
                            n_inner=HIDDEN_SIZE*4)
    byte_config = GPT2Config(vocab_size=256+1,
                            n_positions=PATCH_SIZE+1,
                            n_embd=HIDDEN_SIZE,
                            n_layer=BYTE_NUM_LAYERS,
                            n_head=HIDDEN_SIZE//64,
                            n_inner=HIDDEN_SIZE*4)
    model = bGPTLMHeadModel(patch_config, byte_config, PATCH_SIZE, PATCH_SAMPLING_BATCH_SIZE,EMBEDDING_CHANGE)
    model = model.to(device)
    model.eval()

    # load filenames under train and eval folder
    eval_files = list_files_in_directory(EVAL_FOLDERS)
    print(f"Number of files in eval folder: {len(eval_files)}")
    print("the eval_folders are: ", EVAL_FOLDERS)

    eval_batch_nums = int(len(eval_files) / BATCH_SIZE)

    random.shuffle(eval_files)

    eval_files = eval_files[:eval_batch_nums*BATCH_SIZE]

    eval_set = ByteDataset(eval_files, PATCH_SIZE, PATCH_LENGTH, None, 'eval')

    eval_set = DataLoader(eval_set, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle = False)


    if LOAD_FROM_PRETRAINED and os.path.exists(PRETRAINED_PATH):

        # Load checkpoint to CPU then to GPU
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
        cpu_model = deepcopy(model)
        cpu_model.load_state_dict(checkpoint['model'])
        model.load_state_dict(cpu_model.state_dict())

    eval_loss = eval_epoch(eval_set, model)
    
    print(f"Bits per byte loss: {eval_loss} for embedding change {EMBEDDING_CHANGE} on {EVAL_FOLDERS}")
    
    # clear gpu memory
    torch.cuda.empty_cache()
            
    





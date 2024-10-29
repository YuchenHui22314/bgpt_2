import os
import time
import wandb
import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader

def collate_batch(input_batches):

    # a = zip(*[(1, 4), (2, 5), (3, 6)]) 
    # print(list(a)) # [(1, 2, 3), (4, 5, 6)]

    input_patches, input_masks = zip(*input_batches)
    # pad to the same length (max length in the batch)
    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=256)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True, padding_value=0)

    return input_patches, input_masks

def split_into_minibatches(input_patches, input_masks, minibatch_size):
    minibatches = []
    for start_idx in range(0, len(input_patches), minibatch_size):
        end_idx = start_idx + minibatch_size
        minibatch_patches = input_patches[start_idx:end_idx]
        minibatch_masks = input_masks[start_idx:end_idx]
        minibatches.append((minibatch_patches, minibatch_masks))
    return minibatches

def list_files_in_directory(directories):
    file_list = []
    
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

def read_bytes(filename, PATCH_SIZE, PATCH_LENGTH, SHOW_WARNS):
    
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
        if SHOW_WARNS:
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

class ByteDataset(Dataset):
    def __init__(self, filenames, PATCH_SIZE, PATCH_LENGTH, CONVERSION_MODE, split='train'):
        self.PATCH_SIZE = PATCH_SIZE
        self.PATCH_LENGTH = PATCH_LENGTH
        self.CONVERSION_MODE = CONVERSION_MODE
        ############################################
        # construct filenames pairs and extensions
        ############################################
        if CONVERSION_MODE == None:
            print(f"Autoregressive Training Mode: loading {len(filenames)} files for {split}")
            self.filenames = filenames
        elif "->" in CONVERSION_MODE:
            print(f"Unidirectional Conversion Mode: loading {len(filenames)} files for {split}")
            input_ext = CONVERSION_MODE.split("->")[0]
            target_ext = CONVERSION_MODE.split("->")[1]

            self.filenames = []
            for filename in filenames:
                if filename.split('.')[-1]==input_ext:
                    target_filename = filename[:-(len(input_ext))] + target_ext
                    if os.path.exists(target_filename):
                        self.filenames.append((filename, target_filename))
        elif "&" in CONVERSION_MODE:
            print(f"Bidirectional Conversion Mode: loading {len(filenames)} files for {split}")
            input_ext = CONVERSION_MODE.split("&")[0]
            target_ext = CONVERSION_MODE.split("&")[1]

            # if &, then each file appears twice, once as input and once as target.
            self.filenames = []
            for filename in filenames:
                if filename.split('.')[-1]==input_ext:
                    target_filename = filename[:-(len(input_ext))] + target_ext
                    if os.path.exists(target_filename):
                        self.filenames.append((filename, target_filename))
                elif filename.split('.')[-1]==target_ext:
                    input_filename = filename[:-(len(target_ext))] + input_ext
                    if os.path.exists(input_filename):
                        self.filenames.append((input_filename, filename))
        else:
            raise ValueError("Invalid Conversion Mode, please check the config.py file. You can use None, 'input->output', or 'input&output'.")
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        ############################################
        # read bytes from file
        ############################################
        if self.CONVERSION_MODE == None:
            filename = self.filenames[idx]
            file_bytes, file_masks = read_bytes(filename, self.PATCH_SIZE, self.PATCH_LENGTH, False)
        else:
            input_filename, target_filename = self.filenames[idx]
            # NOTE Here the masks are 1s of patch sequence length.
            input_bytes, input_masks = read_bytes(input_filename, self.PATCH_SIZE, self.PATCH_LENGTH, False)
            target_bytes, target_masks = read_bytes(target_filename, self.PATCH_SIZE, self.PATCH_LENGTH, False)

            # NOTE -patch_size means that the last eos patch is removed.
            # and we will add the target extension instead 
            file_bytes = input_bytes[:-self.PATCH_SIZE] + target_bytes
            file_masks = input_masks[:-1] + target_masks

            if len(file_bytes) > self.PATCH_LENGTH*self.PATCH_SIZE:
                print(f"Warning: {input_filename} and {target_filename} are too long after concatenation, truncating to {self.PATCH_LENGTH*self.PATCH_SIZE} bytes.")
                # TODO: PROBLEM: if we can not just truncate... What if all target extensions are lost? Really have to determine How they train all this.... May be restrict the size of the files in the "train" folder? TBD.
                file_bytes = file_bytes[:self.PATCH_LENGTH*self.PATCH_SIZE]
                file_masks = file_masks[:self.PATCH_LENGTH]

        file_bytes = torch.tensor(file_bytes, dtype=torch.long)
        file_masks = torch.tensor(file_masks, dtype=torch.long)
        
        return file_bytes, file_masks
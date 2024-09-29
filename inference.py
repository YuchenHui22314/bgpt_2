import os
import time
import torch
from utils import *
from config import *
from transformers import  GPT2Config

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                    max_length=PATCH_LENGTH, 
                    max_position_embeddings=PATCH_LENGTH,
                    hidden_size=HIDDEN_SIZE,
                    n_head=HIDDEN_SIZE//64,
                    vocab_size=1)   # vocal size is not important here.
byte_config = GPT2Config(num_hidden_layers=BYTE_NUM_LAYERS, 
                    max_length=PATCH_SIZE+1, 
                    max_position_embeddings=PATCH_SIZE+1,
                    hidden_size=HIDDEN_SIZE,
                    n_head=HIDDEN_SIZE//64,
                    vocab_size=256+1) # vocal size is all possible values of a byte plus 1 for eos.

model = bGPTLMHeadModel(patch_config, byte_config)
print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

checkpoint = torch.load(INFERENCE_WEIGHTS_PATH, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

# function to read input bytes from input file.
def read_bytes(filename):
    
    # transfer the extension name to bytes.
    # why truncate the extension to PATCH_SIZE?
    # but in fact, the extension is never truncated, because we use 16 as the default patch size.
    # and the extension is always shorter than 16.
    ext = filename.split('.')[-1]
    ext = bytearray(ext, 'utf-8')
    ext = [byte for byte in ext][:PATCH_SIZE]

    # read raw bytes from file.
    # TODO: check if it is utf-8
    with open(filename, 'rb') as f:
        file_bytes = f.read()

    # after this step, bytes is a list of integers.
    bytes = []
    for byte in file_bytes:
        bytes.append(byte)

    # add "end of patch" byte with id 256 
    # (the 257 th possible value), 
    # given other bytes have possible values of 0-255.
    if len(bytes)%PATCH_SIZE!=0:
        bytes = bytes + [256] * (PATCH_SIZE - len(bytes) % PATCH_SIZE)

    # bos = beginning of sequence (of patches)
    # if len(ext) > PATCH_SIZE, wll not add 256.
    # trying to add file extension info to the beginning of the sequence.
    bos_patch = ext + [256] * (PATCH_SIZE - len(ext))
    # why add eos at the last patch? because during training they add eos at the end of 
    # every sequence. TODO: check the training code.
    bytes = bos_patch + bytes + [256] * PATCH_SIZE

    # not sure if we will truncate sth here. Ideally not.
    bytes = bytes[:PATCH_LENGTH*PATCH_SIZE]

    return bytes

# byte stream for target extension. (why no truncation?)
bos_patch = [byte for byte in bytearray(TARGET_EXT, 'utf-8')]
bos_patch = bos_patch + [256] * (PATCH_SIZE - len(bos_patch))

# all file names to convert
if INFERENCE_MODE == "convert":
    files = os.listdir(INPUT_FOLDER)
    files = [i for i in files if i.split('.')[-1] == INPUT_EXT]
else:
    files = list(range(NUM_SAMPLES))

for i in files:
    if INFERENCE_MODE == "convert":
        # output file name
        filename = OUTPUT_FOLDER+"/"+i+'.'+TARGET_EXT
        # we have to remove the last patch, filling with eos.
        # then add the extension info to the beginning of the sequence.
        byte_list = read_bytes(INPUT_FOLDER+"/"+i)[:-PATCH_SIZE]+bos_patch
    else:
        # generate 100 samples and properly name them.
        filename = OUTPUT_FOLDER+"/"+time.strftime("%Y%m%d-%H%M%S")+"-"+str(i+1)+"."+TARGET_EXT
        # pure generation without input? interesting.
        byte_list = bos_patch.copy()

    prefix_len = len(byte_list)

    # batch size is 1
    input_patches = torch.tensor([byte_list], device=device)
    while input_patches.shape[1]<PATCH_LENGTH*PATCH_SIZE:
        predicted_patch = model.generate(input_patches.unsqueeze(0),
                                         top_k=TOP_K,
                                         top_p=TOP_P,
                                         temperature=TEMPERATURE)
        for byte in predicted_patch:
            if byte == 256:
                break
            byte_list.append(byte)
        if byte == 256:
            break
        predicted_patch = torch.tensor([predicted_patch], device=device)
        input_patches = torch.cat([input_patches, predicted_patch], dim=1)

    # only take output bytes, discard input bytes.
    byte_list = byte_list[prefix_len:]

    # set output file name as the current time
    with open(filename, 'wb') as file:
        for byte in byte_list:
            file.write(bytes([byte]))
        if INFERENCE_MODE == "convert":
            print("Converted to "+filename)
        else:
            print("Generated "+filename)

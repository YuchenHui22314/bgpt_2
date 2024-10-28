import os
import time
import wandb
import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, get_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import bGPTLMHeadModel
from data import *
from config_test import *

#############################
# Set up distributed training
#############################

# world size, rank and local rank are set by commands
# such as torchrun.
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

if world_size > 1:
    torch.cuda.set_device(local_rank)   
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl') if world_size > 1 else None
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if DETERMINISTIC:
    # Set random seed
    seed = 0 + global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

batch_size = BATCH_SIZE

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

model = bGPTLMHeadModel(patch_config, byte_config, PATCH_SIZE, PATCH_SAMPLING_BATCH_SIZE,"none")
model = model.to(device)

# print parameter number
print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

scaler = GradScaler()
is_autocast = True
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    

# call model with a batch of input
def process_one_batch(batch):
    input_patches, input_masks = batch
    loss = model(input_patches, input_masks).loss

    # Reduce the loss on GPU 0
    if world_size > 1:
        loss = loss.unsqueeze(0)
        # add all the losses together and broadcast to GPU 0 
        # NOTE: This code is redundant. The gradient on a GPU is only non-0 when using the loss on that GPU. So useless to reduce the loss on GPU 0.
        # FIXME should remove / world_size. Because will do so automatically in .backward()
        dist.reduce(loss, dst=0)
        #loss = loss / world_size
        dist.broadcast(loss, src=0)

    return loss

# do one epoch for training
def train_epoch():
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()
    train_steps = (epoch-1)*len(train_set)

    for batch in tqdm_train_set:
        minibatches = split_into_minibatches(batch[0], batch[1], BATCH_SIZE//ACCUMULATION_STEPS)
        # Accumulate gradients for ACCUMULATION_STEPS minibatches
        for minibatch in minibatches:
            with autocast():
                loss = process_one_batch(minibatch) / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            total_train_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        
        lr_scheduler.step()
        # set to none = True can save memory
        # here it is equivalent to model.zero_grad()
        model.zero_grad(set_to_none=True)
        tqdm_train_set.set_postfix({str(global_rank)+'_train_loss': total_train_loss / iter_idx})
        train_steps += 1
        
        # Log the training loss to wandb
        if global_rank==0 and WANDB_LOG:
            wandb.log({"train_loss": total_train_loss / iter_idx}, step=train_steps)

        iter_idx += 1
        
    return total_train_loss / (iter_idx-1)

# do one epoch for eval
def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set: 
        minibatches = split_into_minibatches(batch[0], batch[1], BATCH_SIZE//ACCUMULATION_STEPS)
        for minibatch in minibatches:
            with torch.no_grad():
                loss = process_one_batch(minibatch) / ACCUMULATION_STEPS
            total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({str(global_rank)+'_eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1
    return total_eval_loss / (iter_idx-1)

# train and eval
if __name__ == "__main__":
    
    if global_rank==0 and WANDB_LOG:
        # Initialize wandb
        wandb.init(project="bgpt", name="midi_gen_p_size_"+str(PATCH_SIZE)+
                    "_p_length_"+str(PATCH_LENGTH)+
                    "_b_layers_"+str(BYTE_NUM_LAYERS)+
                    "_p_layers_"+str(PATCH_NUM_LAYERS)+
                    "_h_size_"+str(HIDDEN_SIZE)+
                    "_lr_"+str(LEARNING_RATE)+
                    "_batch_"+str(BATCH_SIZE))
                   
    # load filenames under train and eval folder
    train_files = list_files_in_directory(TRAIN_FOLDERS)
    eval_files = list_files_in_directory(EVAL_FOLDERS)

    if len(eval_files)==0:
        random.shuffle(train_files)
        eval_files = train_files[:int(len(train_files)*EVAL_SPLIT)]
        train_files = train_files[int(len(train_files)*EVAL_SPLIT):]
        
    train_batch_nums = int(len(train_files) / batch_size)
    eval_batch_nums = int(len(eval_files) / batch_size)

    random.shuffle(train_files)
    random.shuffle(eval_files)

    train_files = train_files[:train_batch_nums*batch_size]
    eval_files = eval_files[:eval_batch_nums*batch_size]

    train_set = ByteDataset(train_files, split='train')
    eval_set = ByteDataset(eval_files, split='eval')

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=local_rank)

    train_set = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_batch, sampler=train_sampler, shuffle = (train_sampler is None))
    eval_set = DataLoader(eval_set, batch_size=batch_size, collate_fn=collate_batch, sampler=eval_sampler, shuffle = (train_sampler is None))

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=NUM_EPOCHS * len(train_set) // 10,
        num_training_steps=NUM_EPOCHS * len(train_set),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LOAD_FROM_PRETRAINED and os.path.exists(PRETRAINED_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')

        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
            
        print(f"Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Loss {checkpoint['min_eval_loss']}")
    
    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = float('inf')

    if LOAD_FROM_CHECKPOINT and os.path.exists(WEIGHTS_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')

        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])
        pre_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        min_eval_loss = checkpoint['min_eval_loss']
        print("Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)
        checkpoint = None
    
    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = float('inf')

    for epoch in range(1+pre_epoch, NUM_EPOCHS+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        train_loss = train_epoch()
        eval_loss = eval_epoch()
        if global_rank==0:
            with open(LOGS_PATH,'a') as f:
                f.write("Epoch " + str(epoch) + "\ntrain_loss: " + str(train_loss) + "\neval_loss: " +str(eval_loss) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
            if eval_loss < min_eval_loss:
                best_epoch = epoch
                min_eval_loss = eval_loss
                checkpoint = { 
                                'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_sched': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                'min_eval_loss': min_eval_loss
                                }
                torch.save(checkpoint, WEIGHTS_PATH[:-4] + f"_E{epoch}.pth")
        
        if world_size > 1:
            # wait for all processes to finish one epoch
            dist.barrier()

    if global_rank==0:
        print("Best Eval Epoch : "+str(best_epoch))
        print("Min Eval Loss : "+str(min_eval_loss))


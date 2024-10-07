import torch
import random
from config import *
from transformers import GPT2Model, GPT2LMHeadModel, PreTrainedModel
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PatchLevelDecoder(PreTrainedModel):
    """
    A Patch-level Decoder model for generating patch features in an auto-regressive manner. 
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE * (256+1), config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = GPT2Model(config)

    def forward(self,
                patches: torch.Tensor,
                masks=None) -> torch.Tensor:
        """
        The forward pass of the patch-level decoder model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the encoded patches
        """
        patches = torch.nn.functional.one_hot(patches, num_classes=256+1).to(self.dtype)
        patches = patches.reshape(len(patches), -1, PATCH_SIZE * (256+1))
        patches = self.patch_embedding(patches.to(self.device))

        if masks==None:
            return self.base(inputs_embeds=patches)
        else:
            return self.base(inputs_embeds=patches,
                             attention_mask=masks)

class ByteLevelDecoder(PreTrainedModel):
    """
    A Byte-level Decoder model for generating the bytes within each patch in an auto-regressive manner
    based on the encoded patch features. It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.special_token_id = 256
        self.base = GPT2LMHeadModel(config)

    def forward(self,
                encoded_patches: torch.Tensor,
                target_patches: torch.Tensor):
        """
        The forward pass of the byte-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the output of the model
        """
        # preparing the labels for model training
        target_patches = torch.cat((torch.ones_like(target_patches[:,0:1])*self.special_token_id, target_patches), dim=1)

        # select patches
        if PATCH_SAMPLING_BATCH_SIZE!=0 and PATCH_SAMPLING_BATCH_SIZE<target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:PATCH_SAMPLING_BATCH_SIZE])

            target_patches = target_patches[selected_indices,:]
            encoded_patches = encoded_patches[selected_indices,:]

        # get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)

        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:,1:,:]), dim=1)

        return self.base(inputs_embeds=inputs_embeds,
                         labels=target_patches)

    def generate(self,
                 encoded_patch: torch.Tensor,
                 tokens: torch.Tensor):
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability distribution of next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:,1:,:]), dim=1)
        
        # Get output from model
        outputs = self.base(inputs_embeds=tokens)
        
        # Get probabilities of next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs

class bGPTLMHeadModel(PreTrainedModel):
    """
    bGPT is a byte-level language model with a hierarchical structure.
    It includes a patch-level decoder and a byte-level decoder.
    The patch-level decoder is used to generate patch features in an auto-regressive manner.
    The byte-level decoder is used to generate the bytes within each patch in an auto-regressive manner.
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.special_token_id = 256
        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.byte_level_decoder = ByteLevelDecoder(decoder_config)

    def forward(self,
                patches: torch.Tensor,
                masks: torch.Tensor):
        """
        The forward pass of the bGPT model.
        :param patches: the patches to be encoded, size: (batch_size, byte_sequence_length)
        :param masks: the masks for the patches
        :return: the decoded patches
        """
        # split byte sequence into patches
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        
        # patch decoder output
        encoded_patches = self.patch_level_decoder(patches, masks)["last_hidden_state"]
        
        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks[:, 0] = 0
        
        encoded_patches = encoded_patches[left_shift_masks == 1]
        patches = patches[masks == 1]
        
        return self.byte_level_decoder(encoded_patches, patches)
        
    def generate(self,
                 patches: torch.Tensor,
                 top_k=0,
                 top_p=1,
                 temperature=1.0):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :param top_k: the top k for sampling
        :param top_p: the top p for sampling
        :param temperature: the temperature for sampling
        :return: the generated patches
        """
        if patches.shape[-1]%PATCH_SIZE!=0:
            tokens = patches[:,:,-(patches.shape[-1]%PATCH_SIZE):].squeeze(0).squeeze(0)
            tokens = torch.cat((torch.tensor([self.special_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:,:,:-(patches.shape[-1]%PATCH_SIZE)]
        else:
            tokens = torch.tensor([self.special_token_id], device=self.device)
            
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        generated_patch = []            

        while True:
            prob = self.byte_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            token = temperature_sampling(prob, temperature=temperature)
            generated_patch.append(token)
            if token == self.special_token_id or len(tokens) >= PATCH_SIZE:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch

class bGPTForClassification(PreTrainedModel):
    """
    This class is used to classify the patches generated by the bGPT model.
    It contains the patch level decoder and a classifier.
    The global average pooling is used to get the patch-level representation.
    Then, the patch-level representation is used to classify the patches.
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, encoder_config, label_size):
        super().__init__(encoder_config)
        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.classifier = torch.nn.Linear(encoder_config.n_embd, label_size)
        torch.nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self,
                patches: torch.Tensor):
        """
        The forward pass of the bGPT model for classification.
        :param patches: the patches to be both encoded and decoded
        :return: the logits generated by the classifier
        """
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        encoded_patches = torch.mean(encoded_patches, dim=1)
        return self.classifier(encoded_patches)

##########################################################
# Byte Frequency Distribution
##########################################################

# code inspired by https://github.com/SCPJaehyeon/byte-frequency-distribution-processing-module

class ByteFrequencyDistribution():
    """Get byte frequency distribution table & shannon entropy results."""

    def get_fingerprint_by_folder(self, folder_path):
        """Get byte frequency distribution table & shannon entropy results.
        @param foler_path: folder path.
        @return: fingerprint dict or None.
        """

        file_list = os.listdir(folder_path)
        print(file_list)
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
        df = df.loc[:, (df != 0).any(axis=0)]
        # calculate correlation matrix
        corr = df.corr()
        print(corr.shape)
        return corr, average_byte_frequency_table






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


    def _get_shannon_entropy(self, bytefreq):
        """Calculate shannon entropy(min bits per byte-character).
        @param bytefreq: byte frequency list.
        @return: shannon entropy float.
        """
        ent = 0.0
        for freq in bytefreq:
            freq = freq / 100
            if freq > 0:
                ent = ent + freq * math.log(freq, 2)
        ent = -ent
        return round(ent, 3)

    def run(self, file_path):
        """Run byte frequency distribution analysis.
        @return: analysis results dict or None.
        """
        if not os.path.exists(self.file_path):
            return {}

        try:
            frequency_table, _ = self.get_byte_frequency_table_by_file_path(file_path)
            if frequency_table:
                frequency_table["shannon_entropy"] = self._get_shannon_entropy(frequency_table.values())
            return frequency_table
        except Exception as e:
            print(exception(e))
            return {}
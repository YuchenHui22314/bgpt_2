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
        # map each patch to a vecor of size n_embd
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
        # map each byte in the patch to a one-hot vector of length 256+1
        patches = torch.nn.functional.one_hot(patches, num_classes=256+1).to(self.dtype)
        # concat these one-hot vectors to get a vector of size PATCH_SIZE * (256+1)
        patches = patches.reshape(len(patches), -1, PATCH_SIZE * (256+1))
        # map this vector to a vector of size n_embd
        patches = self.patch_embedding(patches.to(self.device))

        # Eliminating padding bytes 
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
        # GPT2LMHeadMoel is GPT2 + a language model head, which maps the last hidden state to the output logits of length vocab_size
        self.base = GPT2LMHeadModel(config)

    def forward(self,
                encoded_patches: torch.Tensor,
                target_patches: torch.Tensor):
        """
        The forward pass of the byte-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the output of the model
        Shape of encoded_patch: [# of non-masked patches, hidden_size]
        Shape of target patches: [# of non-masked patches, patch_size]
        """
        
        # In each patch, insert a special byte at the beginning.
        # example: [1, 2, 3, 4] -> [256, 1, 2, 3, 4]
        target_patches = torch.cat((torch.ones_like(target_patches[:,0:1])*self.special_token_id, target_patches), dim=1)

        # Random selection of patches
        if PATCH_SAMPLING_BATCH_SIZE!=0 and PATCH_SAMPLING_BATCH_SIZE<target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:PATCH_SAMPLING_BATCH_SIZE])

            target_patches = target_patches[selected_indices,:]
            encoded_patches = encoded_patches[selected_indices,:]

        # get input embeddings
        # map byte values to embeddings, shape: [# of non-padding patches in all batches, patch_size, hidden_size]
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)

        # concatenate the encoded patches with the input embeddings. Replace the first special byte we just added at the beginning of each patch with the encoded patch.

        # shape: [# of non-padding patches in all batches, patch_size+1, hidden_size]
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:,1:,:]), dim=1)

        # NOTE (1) automatically adapted labels. for example, if the input is [1, 2, 3, 4], the label should also be [1, 2, 3, 4] and finally only [2,3,4] will be used to calculate the loss. So here, even if the first byte is a special token in target_patches, it is still valid. Proof in source code: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/gpt2/modeling_gpt2.py#L1341
        # NOTE (2) Here, batch size becomes # of non-padding patches in all batches, and sequence length becomes patch_size+1, which is drastically reduced. That is why it can save computation.
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
        '''
        Shape of encoded_patch: [# of non-masked patches, hidden_size]
        Shape of token: [# of non-masked patches, patch_size]
        '''
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
        :param masks: the masks for the patches(1 for content, 0 for padding), size: (batch_size, patch_sequence_length)
        :return: the decoded patches
        """
        # split byte sequence into patches
        # shape of patches: [batch_size, patch_sequence_length, patch_size]
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        
        # patch decoder output
        # [batch_size, patch_sequence_length, hidden_size]
        encoded_patches = self.patch_level_decoder(patches, masks)["last_hidden_state"]
        
        '''
        mask = 
        [[1, 1, 1, 0, 0]
        [1, 1, 0, 0, 0]]
        '''
        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        '''
        left_shift_masks = 
        [[1, 1, 0, 0, 0]
        [1, 0, 0, 0, 0]]
        '''
        masks[:, 0] = 0
        
        '''
        mask = 
        [[0, 1, 1, 0, 0]
        [0, 1, 0, 0, 0]]
        '''
        

        # shape of encoded_patches: [number of patches with index left_shift_masks == 1., hidden_size]. So it is in fact flattened.
        # So. the objective of this step:
        # (1) remove the last patch for encoded_patches because we cannot insert this to the next bytes in a patch (there is no next patch...)(not nessarily the eos patch, but still designed for this purpose)
        # (2) remove the first patch for patches, because there is no patch encoding for this patch to insert at position 0 (not necessarily the bos patch, but still designed for this purpose)
        # (3) not necessarily because in read_bytes(filename), there is a random slicing algorithm.
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


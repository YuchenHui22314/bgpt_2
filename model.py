import torch
import random
from transformers import GPT2Model, GPT2LMHeadModel, PreTrainedModel
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
import numpy as np
from sklearn.preprocessing import minmax_scale
import os

from utils import *

def outlier_handler(numpy_array, how_to_handle):
    if how_to_handle == "none":
        return numpy_array
    elif how_to_handle == "clip":
        numpy_array = np.clip(numpy_array, -1, 1)
        return numpy_array
    elif how_to_handle == "z-normalization":
        numpy_array = zscore(numpy_array, axis=1)
    elif how_to_handle == "min-max":
        numpy_array = minmax_scale(numpy_array, axis=1)
    elif how_to_handle == "max":
        max_value = np.max(numpy_array, axis=1)
        # make it same shape as the input
        numpy_array = np.repeat(max_value[:, np.newaxis], numpy_array.shape[1], axis=1)
    elif how_to_handle == "min":
        min_value = np.min(numpy_array, axis=1)
        # make it same shape as the input
        numpy_array = np.repeat(min_value[:, np.newaxis], numpy_array.shape[1], axis=1)

    return numpy_array

def outlier_handler_tensor(tensor, how_to_handle):
    if how_to_handle == "none":
        return tensor
    elif how_to_handle == "clip":
        tensor = torch.clamp(tensor, -1, 1)
    elif how_to_handle == "z-normalization":
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        tensor = (tensor - mean) / std
    elif how_to_handle == "min-max":
        min = tensor.min(dim=1, keepdim=True).values
        max = tensor.max(dim=1, keepdim=True).values
        tensor = (tensor - min) / (max - min)
    elif how_to_handle == "max":
        tensor = tensor.max(dim=1, keepdim=True).values

    return tensor

class PatchLevelDecoder(PreTrainedModel):
    """
    A Patch-level Decoder model for generating patch features in an auto-regressive manner. 
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config, PATCH_SIZE):
        super().__init__(config)
        # map each patch to a vecor of size n_embd
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE * (256+1), config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = GPT2Model(config)
        self.PATCH_SIZE = PATCH_SIZE

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
        patches = patches.reshape(len(patches), -1, self.PATCH_SIZE * (256+1))
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
    def __init__(self, config, PATCH_SIZE, PATCH_SAMPLING_BATCH_SIZE):
        super().__init__(config)
        self.special_token_id = 256
        # GPT2LMHeadMoel is GPT2 + a language model head, which maps the last hidden state to the output logits of length vocab_size
        self.base = GPT2LMHeadModel(config)
        self.PATCH_SIZE = PATCH_SIZE
        self.PATCH_SAMPLING_BATCH_SIZE = PATCH_SAMPLING_BATCH_SIZE

    def forward(self,
                encoded_patches: torch.Tensor,
                target_patches: torch.Tensor):
        """
        The forward pass of the byte-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target bytes.
        :return: the output of the model
        Shape of encoded_patch: [# of non-masked patches, hidden_size]
        Shape of target patches: [# of non-masked patches, patch_size] (every element is a byte value)
        """
        
        # In each patch, insert a special byte at the beginning.
        # example: [1, 2, 3, 4] -> [256, 1, 2, 3, 4]
        target_patches = torch.cat((torch.ones_like(target_patches[:,0:1])*self.special_token_id, target_patches), dim=1)

        # Random selection of patches
        if self.PATCH_SAMPLING_BATCH_SIZE!=0 and self.PATCH_SAMPLING_BATCH_SIZE<target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:self.PATCH_SAMPLING_BATCH_SIZE])

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
    def __init__(self, encoder_config, decoder_config, PATCH_SIZE, PATCH_SAMPLING_BATCH_SIZE, embedding_process):
        super().__init__(encoder_config)
        self.special_token_id = 256
        self.patch_level_decoder = PatchLevelDecoder(encoder_config, PATCH_SIZE)
        self.byte_level_decoder = ByteLevelDecoder(decoder_config, PATCH_SIZE, PATCH_SAMPLING_BATCH_SIZE)
        self.PATCH_SIZE = PATCH_SIZE
        self.embedding_process = embedding_process

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
        # shape of patches after this operation: 
        # [batch_size, patch_sequence_length, patch_size]
        patches = patches.reshape(len(patches), -1, self.PATCH_SIZE)
        
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
        encoded_patches = outlier_handler_tensor(encoded_patches, self.embedding_process)
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
        if patches.shape[-1]%self.PATCH_SIZE!=0:
            tokens = patches[:,:,-(patches.shape[-1]%self.PATCH_SIZE):].squeeze(0).squeeze(0)
            tokens = torch.cat((torch.tensor([self.special_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:,:,:-(patches.shape[-1]%self.PATCH_SIZE)]
        else:
            tokens = torch.tensor([self.special_token_id], device=self.device)
            
        patches = patches.reshape(len(patches), -1, self.PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        generated_patch = []            

        while True:
            prob = self.byte_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            token = temperature_sampling(prob, temperature=temperature)
            generated_patch.append(token)
            if token == self.special_token_id or len(tokens) >= self.PATCH_SIZE:
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
    def __init__(self, encoder_config, label_size, PATCH_SIZE):
        super().__init__(encoder_config)
        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.classifier = torch.nn.Linear(encoder_config.n_embd, label_size)
        torch.nn.init.normal_(self.classifier.weight, std=0.02)
        self.PATCH_SIZE = PATCH_SIZE

    def forward(self,
                patches: torch.Tensor):
        """
        The forward pass of the bGPT model for classification.
        :param patches: the patches to be both encoded and decoded
        :return: the logits generated by the classifier
        """
        patches = patches.reshape(len(patches), -1, self.PATCH_SIZE)
        
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        encoded_patches = torch.mean(encoded_patches, dim=1)
        return self.classifier(encoded_patches)
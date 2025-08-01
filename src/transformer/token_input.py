from device_manager import DeviceManager
from typing_helpers import ArrayType, List
from tokenizers import Tokenizer 
from tokenizers import bpe_tokenizer as bpe

import numpy as np

class TokenInputLayer:
    def __init__(self, tokenizer:Tokenizer, device_manager:DeviceManager=None):
        self.tokenizer = tokenizer
        self.device_manager = device_manager
    
    def activate(self, batch:List[str]):

        token_ids = [self.tokenizer.encode_to_ids(seq) for seq in batch]
        max_len = max([len(seq) for seq in batch])
        pad_id = self.tokenizer.encode_to_ids('<pad>')
        padded = [self.tokenizer.encode_to_ids(seq) + (max_len - len(seq))*pad_id  for seq in batch]

        return padded



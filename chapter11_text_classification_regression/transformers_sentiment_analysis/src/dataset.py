# -*- coding: utf-8 -*-

import config
import torch

class BERTDataset:
    def __init__(self, review, target):
        """
        :param review: list or numpy array of strings
        :param targets: list or numpy array which is binary
        """
        self.review = review
        self.target = target
        # we fetch max len and tokenizer from config.py
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        
    def __len__(self):
        # this returns the length of dataset
        return len(self.review)
    
    def __getitem__(self, item):
        # for a given item index, return a dictionary
        # of inputs
        review = str(self.review[item])
        review = " ".join(review.split())
        
        # encode_plus comes from hugginface's transformers
        # and exists for all tokenizers they offer
        # it can be used to convert a given string
        # to ids, mask and token type ids which are
        # needed for models like BERT
        # here, review is a string
        inputs = self.tokenizer.encode_plus(
                                        review,
                                        None,
                                        add_special_tokens=True,
                                        max_length=self.max_len,
                                        pad_to_max_length=True,
                                        )
        
        # ids are ids of tokens generated
        # after tokenizing reviews
        ids = inputs["input_ids"]
        
        # mask is 1 where we have input
        # and 0 where we have padding
        mask = inputs["attention_mask"]
        
        # token type ids behave the same way as
        # mask in this specific case
        # in case of two sentences, this is 0
        # for first sentence and 1 for second sentence
        token_type_ids = inputs["token_type_ids"]
        
        # now we return everything
        # note that ids, mask and token_type_ids
        # are all long datatypes and targets is float
        return {
                "ids": torch.tensor(
                            ids, dtype=torch.long
                            ),
                "mask": torch.tensor(
                            mask, dtype=torch.long
                            ),
                "token_type_ids": torch.tensor(
                            token_type_ids, dtype=torch.long
                            ),
                "targets": torch.tensor(
                            self.target[item], dtype=torch.float
                            )
                }

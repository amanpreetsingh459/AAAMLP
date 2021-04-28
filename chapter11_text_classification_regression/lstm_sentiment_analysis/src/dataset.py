# -*- coding: utf-8 -*-

import torch

class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        :param reviews: this is a numpy array
        :param targets: a vector, numpy array
        """
        self.reviews = reviews
        self.target = targets
        
    def __len__(self):
        # returns length of the dataset
        return len(self.reviews)
    
    def __getitem__(self, item):
        # for any given item, which is an int,
        # return review and targets as torch tensor
        # item is the index of the item in concern
        review = self.reviews[item, :]
        target = self.target[item]
        
        return {"review": torch.tensor(review, dtype=torch.long),
                "target": torch.tensor(target, dtype=torch.float)
                }
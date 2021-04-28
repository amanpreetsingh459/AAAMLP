# -*- coding: utf-8 -*-

import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased

from sklearn import model_selection
from sklearn import metrics

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def train():
    # this function trains the model
    # read the training file and fill NaN values with "none"
    # you can also choose to drop NaN values in this
    # specific dataset
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    
    # sentiment = 1 if its positive
    # else sentiment = 0
    dfx.sentiment = dfx.sentiment.apply(
                                    lambda x: 1 if x == "positive" else 0
                                    )
    
    # we split the data into single training
    # and validation fold
    df_train, df_valid = model_selection.train_test_split(
                                            dfx,
                                            test_size=0.1,
                                            random_state=42,
                                            stratify=dfx.sentiment.values
                                            )
    
    # reset index
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    # initialize BERTDataset from dataset.py
    # for training dataset
    train_dataset = dataset.BERTDataset(
                                        review=df_train.review.values,
                                        target=df_train.sentiment.values
                                        )
    
    # create training dataloader
    train_data_loader = torch.utils.data.DataLoader(
                                        train_dataset,
                                        batch_size=config.TRAIN_BATCH_SIZE,
                                        num_workers=4
                                        )
    
    # initialize BERTDataset from dataset.py
    # for validation dataset
    valid_dataset = dataset.BERTDataset(
                                        review=df_valid.review.values,
                                        target=df_valid.sentiment.values
                                        )
    
    # create validation data loader
    valid_data_loader = torch.utils.data.DataLoader(
                                        valid_dataset,
                                        batch_size=config.VALID_BATCH_SIZE,
                                        num_workers=1
                                        )
    
    # initialize the cuda device
    # use cpu if you dont have GPU
    device = torch.device("cuda")
    
    # load model and send it to the device
    model = BERTBaseUncased()
    model.to(device)
    
    # create parameters we want to optimize
    # we generally dont use any decay for bias
    # and weight layers
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
                                {
                                    "params": [
                                            p for n, p in param_optimizer if
                                            not any(nd in n for nd in no_decay)
                                            ],
                                    "weight_decay": 0.001,
                                    },
                                    {
                                        "params": [
                                            p for n, p in param_optimizer if
                                            any(nd in n for nd in no_decay)
                                            ],
                                        "weight_decay": 0.0,
                                    },
                        ]
    
    # calculate the number of training steps
    # this is used by scheduler
    num_train_steps = int(
                        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS
                        )
    
    # AdamW optimizer
    # AdamW is the most widely used optimizer
    # for transformer based networks
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    
    # fetch a scheduler
    # you can also try using reduce lr on plateau
    scheduler = get_linear_schedule_with_warmup(
                                optimizer,
                                num_warmup_steps=0,
                                num_training_steps=num_train_steps
                                )
    
    # if you have multiple GPUs
    # model model to DataParallel
    # to use multiple GPUs
    model = nn.DataParallel(model)
    
    # start training the epochs
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(
                    train_data_loader, model, optimizer, device, scheduler
                    )
        outputs, targets = engine.eval_fn(
                                    valid_data_loader, model, device
                                    )
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy
            
if __name__ == "__main__":
    train()
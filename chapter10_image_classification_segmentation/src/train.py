# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np

import albumentations

import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model import get_model

if __name__ == "__main__":
    # location of train.csv and train_png folder
    # with all the png images
    data_path = "../input/"
    # cuda/cpu device
    device = "cuda"
    # let's train for 10 epochs
    epochs = 10
    # load the dataframe
    df = pd.read_csv(os.path.join(data_path, "train.csv"))
    # fetch all image ids
    images = df.ImageId.values.tolist()
    # a list with image locations
    images = [
            os.path.join(data_path, "train_png", i + ".png") for i in images
            ]
    # binary targets numpy array
    targets = df.target.values
    # fetch out model, we will try both pretrained
    # and non-pretrained weights
    model = get_model(pretrained=True)
    # move model to device
    model.to(device)
    # mean and std values of RGB channels for imagenet dataset
    # we use these pre-calculated values when we use weights
    # from imagenet.
    # when we do not use imagenet weights, we use the mean and
    # standard deviation values of the original dataset
    # please note that this is a separate calculation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # albumentations is an image augmentation library
    # that allows you to do many different types of image
    # augmentations. here, i am using only normalization
    # notice always_apply=True. we always want to apply
    # normalization
    aug = albumentations.Compose(
                        [
                            albumentations.Normalize(
                                mean, std, max_pixel_value=255.0, always_apply=True
                                )
                            ]
                        )
    
    # instead of using kfold, i am using train_test_split
    # with a fixed random state
    train_images, valid_images, train_targets, valid_targets =
                                train_test_split(
                                images, targets, stratify=targets, random_state=42
                                )
                                
    # fetch the ClassificationDataset class
    train_dataset = dataset.ClassificationDataset(
                                image_paths=train_images,
                                203
                                targets=train_targets,
                                resize=(227, 227),
                                augmentations=aug,
                                )
    
    # torch dataloader creates batches of data
    # from classification dataset class
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=16, shuffle=True, num_workers=4
                        )

    # same for validation data
    valid_dataset = dataset.ClassificationDataset(
                                    image_paths=valid_images,
                                    targets=valid_targets,
                                    resize=(227, 227),
                                    augmentations=aug,
                                    )
    
    valid_loader = torch.utils.data.DataLoader(
                        valid_dataset, batch_size=16, shuffle=False, num_workers=4
                        )
    
    # simple Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # train and print auc score for all epochs
    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(
                                            valid_loader, model, device=device
                                            )
        
        roc_auc = metrics.roc_auc_score(valid_targets, predictions)
        
        print(
                f"Epoch={epoch}, Valid ROC AUC={roc_auc}"
                )
        
        
import logging
import os
import shutil

import pandas as pd
import torch
import yaml
from torchvision.utils import save_image

from ludwig.api import LudwigModel
from ludwig.datasets import camseq

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

# set up Python dictionary to hold model training parameters
with open("./config_camseq.yaml") as f:
    config = yaml.safe_load(f.read())

# Define Ludwig model object that drive model training
model = LudwigModel(config, logging_level=logging.INFO)

# load Camseq dataset
df = camseq.load(split=False)

pred_set = df[0:1]  # prediction hold-out 1 image
data_set = df[1:]  # train,test,validate on remaining images

# initiate model training
train_stats, _, output_directory = model.train(  # training statistics  # location for training results saved to disk
    dataset=data_set,
    experiment_name="simple_image_experiment",
    model_name="single_model",
    skip_save_processed_input=True,
)

# print("{}".format(model.model))

# predict
pred_set.reset_index(inplace=True)
pred_out_df, results = model.predict(pred_set)

if not isinstance(pred_out_df, pd.DataFrame):
    pred_out_df = pred_out_df.compute()
pred_out_df["image_path"] = pred_set["image_path"]
pred_out_df["mask_path"] = pred_set["mask_path"]

for index, row in pred_out_df.iterrows():
    pred_mask = torch.from_numpy(row["mask_path_predictions"])
    pred_mask_path = os.path.dirname(os.path.realpath(__file__)) + "/predicted_" + os.path.basename(row["mask_path"])
    print(f"\nSaving predicted mask to {pred_mask_path}")
    if torch.any(pred_mask.gt(1)):
        pred_mask = pred_mask.float() / 255
    save_image(pred_mask, pred_mask_path)
    print("Input image_path:    {}".format(row["image_path"]))
    print("Label mask_path:     {}".format(row["mask_path"]))
    print(f"Predicted mask_path: {pred_mask_path}")

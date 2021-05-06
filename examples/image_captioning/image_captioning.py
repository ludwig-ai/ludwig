# -*- coding: utf-8 -*-
"""
Image Captioning with Ludwig

In this notebook, we will use ludwig to train an image captioning system. 
We be using the Flickr 8k dataset which contains eight thousand images 
with 5 human generated captions per image. 

Our model will consist of a VGG16 pretrained image encoder, 
a vector encoder, and a lstm text decoder.

First, lets import all the modules we will need
"""
import ludwig
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from ludwig.api import LudwigModel
import logging
from tqdm import tqdm

"""
Let's load the dataset which can be done in two lines
using Ludwig's datasets module. This gives us three dataframes
for our training, validation, and test sets
"""
from ludwig.datasets import flickr8k

train, vali, test = flickr8k.load(split=True)

"""
We will now load the pre-trained VGG16 model from keras 
to help us process our images. Since our dataset is relatively small, 
it would be relatively ineffective to train our own image encoder from scratch
"""
from keras.applications import VGG16
from keras import models

modelvgg = VGG16(include_top=True,weights='imagenet')
modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-2].output)

"""
Here we will define a function that will use the pretrained model to extract 
features from the images in our dataset. Every image will be converted into 
a size-4096 vector encoding.
"""
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

def extract_features(dataset, model):
    features = []
    for path in tqdm(dataset['image_path']):
        image = load_img(path, target_size=(224,224,3))
        image = img_to_array(image)
        nimage = preprocess_input(image)
        y_pred = model.predict(nimage.reshape( (1,) + nimage.shape[:3])).flatten()
        y_pred = np.array2string(y_pred, separator=" ")
        features.append(y_pred[1:-1])
    dataset['features'] = features

"""We will be using Ludwig's vector encoders later on to process 
the features extracted by VGG16. These encoders require inputs to be 
strings of whitespace seperated values, so we will be converting the numpy 
vectors to strings using the `np.array2string()` method.

Note that we need to set numpy's print threshold to np.inf so that the the 
strings will not be truncated
"""
np.set_printoptions(threshold=np.inf)
extract_features(train, modelvgg)
extract_features(vali, modelvgg)
extract_features(test, modelvgg)

"""
There seems to be a threading error if we use tensorflow outside of 
ludwig, so this code block helps us reset the tensorflow context to 
prevent that error during model training. 
"""
from tensorflow.python.eager import context

context._context = None
context._create_context()

"""
We define our configuration to use a dense vector encoder and a 
lstm text decoder. Other hyperparameters such as encoder size, 
batch size, and learning rate are also specifed.
"""
config = {
    "input_features": [
        {
            "name": "features",
            "type": "vector",
            "encoder": "dense",
            "layers": [
                {
                    "fc_size": 2000
                }
            ]
        }
    ],
    "output_features": [
        {            
            "name": "caption0",
            "type": "text",
            "level": "word",
            "decoder": "generator",
            "cell_type": "lstm"
        }
    ],
    "training": {
        "batch_size": 16,
        "learning_rate": 0.0001
    }
}

"""
Let's initialize and train a LudwigModel
"""
model = LudwigModel(config, logging_level=logging.INFO)
train_stats, _, _ = model.train(
    training_set=train,
    validation_set=vali,
    test_set=test,
    experiment_name='image_captioning',
    model_name='example',
    skip_save_model=True
)

"""
We can now use our model to make predictions over the test set
"""
predictions, _ = model.predict(dataset=test)

test = test.reset_index(drop=True)
predictions = predictions.reset_index(drop=True)
test_pred = pd.concat([test, predictions], axis=1)


"""
Lets print out all the captions that our model generated for the test set
"""

for index in range(1000):
    arr = test_pred.iloc[index]["caption0_predictions"]
    caption = ""
    for word in arr:
        if word == '<PAD>':
            print(index, caption)
            break
        else:
            caption = caption + word + " "

"""
If you like to see more visualizations and data analysis you can 
check out the .ipynb notebook for more details
"""
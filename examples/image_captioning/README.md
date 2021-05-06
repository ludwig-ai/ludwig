# Image Captioning with Flickr 8K Dataset

This example is based on [Ludwig's image captioning example](https://ludwig-ai.github.io/ludwig-docs/examples/#image-captioning).

Both the .py script and .ipynb notebook discuss:

1. Inital setup
2. Loading the dataset
3. Preprocessing images with a pretrained VGG16 model
4. Defining the configuration
5. Training a LudwigModel
6. Making predictions with trained model on the test set

The .ipynb notebook also goes in-depth on:

- Visualizations to analyze training process
- Analysis of good quality and bad quality captions that are predicted by the model

About the Flickr-8K dataset

- Contains 8096 JPEG images of different sizes
- Each image is provided with 5 possible captions
- Entire dataset is relatively small (about 1 Gb)
- The dataset is provided by the University of Illinois at Urbana-Champaign

Files and folders included in dataset:

- `Flicker8k_Dataset/`: This folder contains all the images
- `Flickr_8k.trainImages.txt`: contains list of image names in the training set
- `Flickr_8k.devImages.txt`: conatins list of image names in the validation set
- `Flickr_8k.testImages.txt`: contains list of image names in the test set
- `Flickr8k.token.txt`: each line contains an image name and caption
- There are other files included such as a rating for each caption but we won't be using those

### References

M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artifical Intellegence Research, Volume 47, pages 853-899
http://www.jair.org/papers/paper3994.html

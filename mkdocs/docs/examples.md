This section contains several examples of how to build models with Ludwig for a variety of tasks.
For each task we show an example dataset and a sample model definition that can be used to train a model from that data.


Text Classification
===

This example shows how to build a text classifier with Ludwig.
It can be performed using the [Reuters-21578](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/reuters-allcats-6.zip) dataset, in particular the version available on [CMU's Text Analytics course website](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/).
Other datasets available on the same webpage, like [OHSUMED](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/ohsumed-allcats-6.zip), is a well-known medical abstracts dataset, and [Epinions.com](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/epinions.zip), a dataset of product reviews, can be used too as the name of the columns is the same.


| text                                                                                             | class       |
|--------------------------------------------------------------------------------------------------|-------------|
| Toronto  Feb 26 - Standard Trustco said it expects earnings in 1987 to increase at least 15...   | earnings    |
| New York  Feb 26 - American Express Co remained silent on market rumors...                       | acquisition |
| BANGKOK  March 25 - Vietnam will resettle 300000 people on state farms known as new economic...  | coffee      |

```
ludwig experiment \
  --data_csv text_classification.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: text
        type: text
        level: word
        encoder: parallel_cnn

output_features:
    -
        name: class
        type: category
```

Named Entity Recognition Tagging
===

| utterance                                                                        | tag                                                             |
|----------------------------------------------------------------------------------|-----------------------------------------------------------------|
| Blade Runner is a 1982 neo-noir science fiction film directed by Ridley Scott    | Movie Movie O O Date O O O O O O Person Person                  |
| Harrison Ford and Rutger Hauer starred in it                                     | Person Person O Person person O O O                             |
| Philip Dick 's novel Do Androids Dream of Electric Sheep ? was published in 1968 | Person Person O O Book Book Book Book Book Book Book O O O Date |

```
ludwig experiment \
  --data_csv sequence_tags.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: utterance
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        reduce_output: null
        preprocessing:
          word_format: space

output_features:
    -
        name: tag
        type: sequence
        decoder: tagger
```


Natural Language Understanding
===

| utterance                      | intent      | slots                             |
|--------------------------------|-------------|-----------------------------------|
| I want a pizza                 | order_food  | O O O B-Food_type                 |
| Book a flight to Boston        | book_flight | O O O O B-City                    |
| Book a flight at 7pm to London | book_flight | O O O O B-Departure_time O B-City |

```
ludwig experiment \
  --data_csv nlu.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: utterance
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        bidirectional: true
        num_layers: 2
        reduce_output: null
        preprocessing:
          word_format: space

output_features:
    -
        name: intent
        type: category
        reduce_input: sum
        num_fc_layers: 1
        fc_size: 64
    -
        name: slots
        type: sequence
        decoder: tagger
```


Machine Translation
===

| english                   | italian                   |
|---------------------------|---------------------------|
| Hello! How are you doing? | Ciao, come stai?          |
| I got promoted today      | Oggi sono stato promosso! |
| Not doing well today      | Oggi non mi sento bene    |

```
ludwig experiment \
  --data_csv translation.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: english
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        reduce_output: null
        preprocessing:
          word_format: english_tokenize

output_features:
    -
        name: italian
        type: text
        level: word
        decoder: generator
        cell_type: lstm
        attention: bahdanau
        loss:
            type: sampled_softmax_cross_entropy
        preprocessing:
          word_format: italian_tokenize

training:
    batch_size: 96
```


Chit-Chat Dialogue Modeling through Sequence2Sequence
===

| user1                     | user2                                      |
|---------------------------|--------------------------------------------|
| Hello! How are you doing? | Doing well, thanks!                        |
| I got promoted today      | Congratulations!                           |
| Not doing well today      | I’m sorry, can I do something to help you? |

```
ludwig experiment \
  --data_csv chitchat.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: user1
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        reduce_output: null

output_features:
    -
        name: user2
        type: text
        level: word
        decoder: generator
        cell_type: lstm
        attention: bahdanau
        loss:
            type: sampled_softmax_cross_entropy

training:
    batch_size: 96
```


Sentiment Analysis
===

| review                          | sentiment |
|---------------------------------|-----------|
| The movie was fantastic!        | positive  |
| Great acting and cinematography | positive  |
| The acting was terrible!        | negative  |

```
ludwig experiment \
  --data_csv sentiment.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: review
        type: text
        level: word
        encoder: parallel_cnn

output_features:
    -
        name: sentiment
        type: category
```


Image Classification
===

| image_path              | class |
|-------------------------|-------|
| images/image_000001.jpg | car   |
| images/image_000002.jpg | dog   |
| images/image_000003.jpg | boat  |

```
ludwig experiment \
  --data_csv image_classification.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn

output_features:
    -
        name: class
        type: category
```

Image Classification (MNIST)
===
This is a complete example of training an image classification model on the MNIST
dataset.

## Download the MNIST dataset.
```
git clone https://github.com/myleott/mnist_png.git
cd mnist_png/
tar -xf mnist_png.tar.gz
cd mnist_png/
```

## Create train and test CSVs.
Open python shell in the same directory and run this:
```
import os
for name in ['training', 'testing']:
    with open('mnist_dataset_{}.csv'.format(name), 'w') as output_file:
        print('=== creating {} dataset ==='.format(name))
        output_file.write('image_path,label\n')
        for i in range(10):
            path = '{}/{}'.format(name, i)
            for file in os.listdir(path):
                if file.endswith(".png"):
                    output_file.write('{},{}\n'.format(os.path.join(path, file), str(i)))

```
Now you should have `mnist_dataset_training.csv` and `mnist_dataset_testing.csv`
containing 60000 and 10000 examples correspondingly and having the following format

| image_path           | label |
|----------------------|-------|
| training/0/16585.png |  0    |
| training/0/24537.png |  0    |
| training/0/25629.png |  0    |

## Train a model.

From the directory where you have virtual environment with ludwig installed:
```
ludwig train \
  --data_train_csv <PATH_TO_MNIST_DATASET_TRAINING_CSV> \
  --data_test_csv <PATH_TO_MNIST_DATASET_TEST_CSV> \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn
        conv_layers:
            -
                num_filters: 32
                filter_size: 3
                pool_size: 2
                pool_stride: 2
            -
                num_filters: 64
                filter_size: 3
                pool_size: 2
                pool_stride: 2
                dropout: true
        fc_layers:
            -
                fc_size: 128
                dropout: true

output_features:
    -
        name: label
        type: category

training:
    dropout_rate: 0.4
```

Image Captioning
===

| image_path                | caption                   |
|---------------------------|---------------------------|
| imagenet/image_000001.jpg | car driving on the street |
| imagenet/image_000002.jpg | dog barking at a cat      |
| imagenet/image_000003.jpg | boat sailing in the ocean |

```
ludwig experiment \
--data_csv image captioning.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn

output_features:
    -
        name: caption
        type: text
        level: word
        decoder: generator
        cell_type: lstm
```


One-shot Learning with Siamese Networks
===

This example can be considered a simple baseline for one-shot learning on the [Omniglot](https://github.com/brendenlake/omniglot) dataset.
The task is, given two images of two handwritten characters, recognize if they are two instances of the same character or not.

| image_path_1                     |   image_path_2                   | similarity |
|----------------------------------|----------------------------------|------------|
| balinese/character01/0108_13.png | balinese/character01/0108_18.png | 1          |
| balinese/character01/0108_13.png | balinese/character08/0115_12.png | 0          |
| balinese/character01/0108_04.png | balinese/character01/0108_08.png | 1          |
| balinese/character01/0108_11.png | balinese/character05/0112_02.png | 0          |

```
ludwig experiment \
--data_csv balinese_characters.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: image_path_1
        type: image
        encoder: stacked_cnn
        preprocessing:
          width: 28
          height: 28
          resize_image: true
    -
        name: image_path_2
        type: image
        encoder: stacked_cnn
        preprocessing:
          width: 28
          height: 28
          resize_image: true
        tied_weights: image_path_1

combiner:
    type: concat
    num_fc_layers: 2
    fc_size: 256

output_features:
    -
        name: similarity
        type: binary
```

Visual Question Answering
===

| image_path              |   question                                | answer |
|-------------------------|-------------------------------------------|--------|
| imdata/image_000001.jpg | Is there snow on the mountains?           | yes    |
| imdata/image_000002.jpg | What color are the wheels                 | blue   |
| imdata/image_000003.jpg | What kind of utensil is in the glass bowl | knife  |


```
ludwig experiment \
--data_csv vqa.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn
    -
        name: question
        type: text
        level: word
        encoder: parallel_cnn

output_features:
    -
        name: answer
        type: text
        level: word
        decoder: generator
        cell_type: lstm
        loss:
            type: sampled_softmax_cross_entropy
```

Spoken Digit Speech Recognition
===

This is a complete example of training an spoken digit speech recognition model on the "MNIST dataset of speech recognition". 

## Download the free spoken digit dataset.

```
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
mkdir speech_recog_digit_data
cp -r free-spoken-digit-dataset/recordings speech_recog_digit_data
cd speech_recog_digit_data
```

## Create an experiment CSV.

```
echo "audio_path","label" >> "spoken_digit.csv"
cd "recordings"
ls | while read -r file_name; do
   audio_path=$(readlink -m "${file_name}")
   label=$(echo ${file_name} | cut -c1)
   echo "${audio_path},${label}" >> "../spoken_digit.csv"
done
cd "../"
```

Now you should have `spoken_digit.csv` containing 2000 examples having the following format

| audio_path                                              |   label                                   |
|---------------------------------------------------------|-------------------------------------------|
| .../speech_recog_digit_data/recordings/0_jackson_0.wav  | 0                                         |
| .../speech_recog_digit_data/recordings/0_jackson_10.wav | 0                                         |
| .../speech_recog_digit_data/recordings/0_jackson_11.wav | 0                                         |
| ...                                                     | ...                                       |
| .../speech_recog_digit_data/recordings/1_jackson_0.wav  | 1                                         |


## Train a model. 

From the directory where you have virtual environment with ludwig installed: 

```
ludwig experiment \
  --data_csv <PATH_TO_SPOKEN_DIGIT_CSV> \
  --model_definition_file model_definition_file.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: audio_path
        type: audio
        encoder: stacked_cnn
        preprocessing:
            audio_feature:
                type: fbank
                window_length_in_s: 0.025
                window_shift_in_s: 0.01
                num_filter_bands: 80
            audio_file_length_limit_in_s: 1.0
            norm: per_file
        reduce_output: concat
        conv_layers:
            -
                num_filters: 16
                filter_size: 6
                pool_size: 4
                pool_stride: 4
                dropout: true
            -
                num_filters: 32
                filter_size: 3
                pool_size: 2
                pool_stride: 2
                dropout: true
        fc_layers:
            -
                fc_size: 64
                dropout: true

output_features:
    -
        name: label
        type: category

training:
    dropout_rate: 0.4
    early_stop: 10
```


Speaker Verification
===

This example describes how to use Ludwig for a simple speaker verification task.
We assume to have the following data with label 0 corresponding to an audio file of an unauthorized voice and
label 1 corresponding to an audio file of an authorized voice.
The sample data looks as follows:

| audio_path                 |   label                                   |
|----------------------------|-------------------------------------------|
| audiodata/audio_000001.wav | 0                                         |
| audiodata/audio_000002.wav | 0                                         |
| audiodata/audio_000003.wav | 1                                         |
| audiodata/audio_000004.wav | 1                                         |

```
ludwig experiment \
--data_csv speaker_verification.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: audio_path
        type: audio
        preprocessing:
            audio_file_length_limit_in_s: 7.0
            audio_feature:
                type: stft
                window_length_in_s: 0.04
                window_shift_in_s: 0.02
        encoder: cnnrnn

output_features:
    -
        name: label
        type: binary
```


Kaggle's Titanic: Predicting survivors
===

This example describes how to use Ludwig to train a model for the
[kaggle competition](https://www.kaggle.com/c/titanic/), on predicting a passenger's probability of surviving the Titanic
disaster. Here's a sample of the data:

| Pclass | Sex    | Age | SibSp | Parch | Fare    | Survived | Embarked |
|--------|--------|-----|-------|-------|---------|----------|----------|
| 3      | male   | 22  | 1     | 0     |  7.2500 | 0        | S        |
| 1      | female | 38  | 1     | 0     | 71.2833 | 1        | C        |
| 3      | female | 26  | 0     | 0     |  7.9250 | 0        | S        |
| 3      | male   | 35  | 0     | 0     |  8.0500 | 0        | S        |

The full data and the column descriptions can be found [here](https://www.kaggle.com/c/titanic/data).

After downloading the data, to train a model on this dataset using Ludwig,
```
ludwig experiment \
  --data_csv <PATH_TO_TITANIC_CSV> \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: Pclass
        type: category
    -
        name: Sex
        type: category
    -
        name: Age
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: SibSp
        type: numerical
    -
        name: Parch
        type: numerical
    -
        name: Fare
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: Embarked
        type: category

output_features:
    -
        name: Survived
        type: binary
```

Better results can be obtained with morerefined feature transformations and preprocessing, but this example has the only aim to show how this type do tasks and data can be used in Ludwig.


Time series forecasting
===

While direct timeseries prediction is a work in progress Ludwig can ingest timeseries input feature data and make numerical predictions. Below is an example of a model trained to forecast timeseries at five different horizons.

| timeseries_data       |   y1  |   y2  |   y3  |   y4  |   y5  |
|-----------------------|-------|-------|-------|-------|-------|
| 15.07 14.89 14.45 ... | 16.92 | 16.67 | 16.48 | 17.00 | 17.02 |
| 14.89 14.45 14.30 ... | 16.67 | 16.48 | 17.00 | 17.02 | 16.48 |
| 14.45 14.3 14.94 ...  | 16.48 | 17.00 | 17.02 | 16.48 | 15.82 |

```
ludwig experiment \
--data_csv timeseries_data.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: timeseries_data
        type: timeseries

output_features:
    -
        name: y1
        type: numerical
    -
        name: y2
        type: numerical
    -
        name: y3
        type: numerical
    -
        name: y4
        type: numerical
    -
        name: y5
        type: numerical
```


Time series forecasting (weather data example)
===

This example illustrates univariate timeseries forecasting using historical temperature data for Los Angeles.

Dowload and unpack historical hourly weather data available on Kaggle
https://www.kaggle.com/selfishgene/historical-hourly-weather-data

Run the following python script to prepare the training dataset:
```
import pandas as pd
from ludwig.utils.data_utils import add_sequence_feature_column

df = pd.read_csv(
    '<PATH_TO_FILE>/temperature.csv',
    usecols=['Los Angeles']
).rename(
    columns={"Los Angeles": "temperature"}
).fillna(method='backfill').fillna(method='ffill')

# normalize
df.temperature = ((df.temperature-df.temperature.mean()) /
                  df.temperature.std())

train_size = int(0.6 * len(df))
vali_size = int(0.2 * len(df))

# train, validation, test split
df['split'] = 0
df.loc[
    (
        (df.index.values >= train_size) &
        (df.index.values < train_size + vali_size)
    ),
    ('split')
] = 1
df.loc[
    df.index.values >= train_size + vali_size,
    ('split')
] = 2

# prepare timeseries input feature colum
# (here we are using 20 preceeding values to predict the target)
add_sequence_feature_column(df, 'temperature', 20)
df.to_csv('<PATH_TO_FILE>/temperature_la.csv')
```

```
ludwig experiment \
--data_csv <PATH_TO_FILE>/temperature_la.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: temperature_feature
        type: timeseries
        encoder: rnn
        embedding_size: 32
        state_size: 32

output_features:
    -
        name: temperature
        type: numerical
```


Movie rating prediction
===

| year | duration  | nominations |  categories        | rating |
|------|-----------|-------------|--------------------|--------|
| 1921 |   3240    |     0       | comedy drama       |  8.4   |
| 1925 |   5700    |     1       | adventure comedy   |  8.3   |
| 1927 |   9180    |     4       | drama comedy scifi |  8.4   |

```
ludwig experiment \
--data_csv movie_ratings.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: year
        type: numerical
    -
        name: duration
        type: numerical
    -
        name: nominations
        type: numerical
    -
        name: categories
        type: set

output_features:
    -
        name: rating
        type: numerical
```


Multi-label classification
===

| image_path              | tags          |
|-------------------------|---------------|
| images/image_000001.jpg | car man       |
| images/image_000002.jpg | happy dog tie |
| images/image_000003.jpg | boat water    |

```
ludwig experiment \
--data_csv image_data.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn

output_features:
    -
        name: tags
        type: set
```


Multi-Task Learning
===

This example is inspired by the classic paper [Natural Language Processing (Almost) from Scratch](https://arxiv.org/abs/1103.0398) by Collobert et al..

| sentence                    | chunks                       | part_of_speech    | named_entities      |
|-----------------------------|------------------------------|-------------------|---------------------|
| San Francisco is very foggy | B-NP I-NP B-VP B-ADJP I-ADJP | NNP NNP VBZ RB JJ | B-Loc I-Loc O O O   |
| My dog likes eating sausage | B-NP I-NP B-VP B-VP B-NP     | PRP NN VBZ VBG NN | O O O O O           |
| Brutus Killed Julius Caesar | B-NP B-VP B-NP I-NP          | NNP VBD NNP NNP   | B-Per O B-Per I-Per |

```
ludwig experiment \
--data_csv nl_data.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
    -
        name: sentence
        type: sequence
        encoder: rnn
        cell: lstm
        bidirectional: true
        reduce_output: null

output_features:
    -
        name: chunks
        type: sequence
        decoder: tagger
    -
        name: part_of_speech
        type: sequence
        decoder: tagger
    -
        name: named_entities
        type: sequence
        decoder: tagger
```

Simple Regression: Fuel Efficiency Prediction
===

This example replicates the Keras example at https://www.tensorflow.org/tutorials/keras/basic_regression to predict the miles per gallon of a car given its characteristics in the [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg) dataset.

|MPG   |Cylinders |Displacement |Horsepower |Weight |Acceleration |ModelYear |Origin |
|------|----------|-------------|-----------|-------|-------------|----------|-------|
|18.0  |8         |307.0        |130.0      |3504.0 |12.0         |70        |1      |
|15.0  |8         |350.0        |165.0      |3693.0 |11.5         |70        |1      |
|18.0  |8         |318.0        |150.0      |3436.0 |11.0         |70        |1      |
|16.0  |8         |304.0        |150.0      |3433.0 |12.0         |70        |1      |

```
ludwig experiment \
--data_csv auto_mpg.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
training:
    batch_size: 32
    epochs: 1000
    early_stop: 50
    learning_rate: 0.001
    optimizer:
        type: rmsprop
input_features:
    -
        name: Cylinders
        type: numerical
    -
        name: Displacement
        type: numerical
    -
        name: Horsepower
        type: numerical
    -
        name: Weight
        type: numerical
    -
        name: Acceleration
        type: numerical
    -
        name: ModelYear
        type: numerical
    -
        name: Origin
        type: category
output_features:
    -
        name: MPG
        type: numerical
        optimizer:
            type: mean_squared_error
        num_fc_layers: 2
        fc_size: 64

```

Binary Classification: Fraud Transactions Identification
===

| transaction_id | card_id | customer_id | customer_zipcode | merchant_id | merchant_name | merchant_category | merchant_zipcode | merchant_country | transaction_amount | authorization_response_code | atm_network_xid | cvv_2_response_xflg | fraud_label |
|----------------|---------|-------------|------------------|-------------|---------------|-------------------|------------------|------------------|--------------------|-----------------------------|-----------------|---------------------|-------------|
| 469483         | 9003    | 1085        | 23039            | 893         | Wright Group  | 7917              | 91323            | GB               | 1962               | C                           | C               | N                   | 0           |
| 926515         | 9009    | 1001        | 32218            | 1011        | Mums Kitchen  | 5813              | 10001            | US               | 1643               | C                           | D               | M                   | 1           |
| 730021         | 9064    | 1174        | 9165             | 916         | Keller        | 7582              | 38332            | DE               | 1184               | D                           | B               | M                   | 0           |

```
ludwig experiment \
--data_csv transactions.csv \
  --model_definition_file model_definition.yaml
```

With `model_definition.yaml`:

```yaml
input_features:
  -
    name: customer_id
    type: category
  -
    name: card_id
    type: category
  -
    name: merchant_id
    type: category
  -
    name: merchant_category
    type: category
  -
    name: merchant_zipcode
    type: category
  -
    name: transaction_amount
    type: numerical
  -
    name: authorization_response_code
    type: category
  -
    name: atm_network_xid
    type: category
  -
    name: cvv_2_response_xflg
    type: category

combiner:
    type: concat
    num_fc_layers: 1
    fc_size: 48

output_features:
  -
    name: fraud_label
    type: binary
```

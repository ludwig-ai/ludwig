import argparse
import sys
import os

import pandas as pd

import tensorflow as tf
from tensorflow.compat.v1 import saved_model

from ludwig.contrib import contrib_command
from ludwig.models.model import load_model_and_definition
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME
from ludwig.constants import FULL

def saved_model_predict(
    saved_model_path = None,
    ludwig_model_path = None,
    data_csv = None,
    output_dir = None,
):
    if saved_model_path == None:
        raise ValueError('saved_model_path is required')

    if ludwig_model_path == None:
        raise ValueError('ludwig_model_path is required')

    if data_csv == None:
        raise ValueError('data_csv is required')

    model, model_definition = load_model_and_definition(ludwig_model_path)

    train_set_metadata_json_fp = os.path.join(
        ludwig_model_path,
        TRAIN_SET_METADATA_FILE_NAME
    )

    dataset, train_set_metadata = preprocess_for_prediction(
        ludwig_model_path,
        FULL,
        data_csv,
        None,
        train_set_metadata_json_fp,
        True,
    )

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            saved_model_path
        )

        predictions = sess.run(
            'class/predictions_class/predictions_class:0',
            feed_dict={
                'text/text:0': dataset.get('text'),
            }
        )

        df = pd.DataFrame(data=predictions, columns=['class'])
        df.to_csv('saved_model_predictions.csv')
    

def cli(sys_argv):
    
    parser = argparse.ArgumentParser(
        description='This script loads a pretrained tensorflow model '
                    'and uses it to predict',
        prog='ludwig saved_model_predict',
        usage='%(prog)s [options]'
    )

    parser.add_argument(
        '-smp',
        '--saved_model_path',
        help='path of a saved model in tensorflow format'
    )

    parser.add_argument(
        '-lmp',
        '--ludwig_model_path',
        help='path of a saved ludwig model'
    )

    parser.add_argument(
        '-dcs',
        '--data_csv',
        help='input data CSV file'
    )

    args = parser.parse_args(sys_argv)
    saved_model_predict(**vars(args))

if __name__ == '__main__':
    contrib_command("saved_model_predict", *sys.argv)
    cli(sys.argv[1:])

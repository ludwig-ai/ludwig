import argparse
import os
import sys

import pandas as pd
import tensorflow as tf

from ludwig.api import LudwigModel
from ludwig.constants import FULL
from ludwig.contrib import contrib_command
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME


def saved_model_predict(
        saved_model_path=None,
        ludwig_model_path=None,
        data_csv=None,
        output_dir=None,
):
    if saved_model_path == None:
        raise ValueError('saved_model_path is required')

    if ludwig_model_path == None:
        raise ValueError('ludwig_model_path is required')

    if data_csv == None:
        raise ValueError('data_csv is required')

    print("Obtain Ludwig predictions")
    ludwig_model = LudwigModel.load(ludwig_model_path)
    ludwig_predictions_df = ludwig_model.predict(data_csv=data_csv)
    ludwig_predictions_df.to_csv('ludwig_predictions.csv', index=False)
    ludwig_weights = ludwig_model.model.collect_weights(
        ['utterance/fc_0/weights:0']
    )['utterance/fc_0/weights:0']
    print(ludwig_weights[0])
    ludwig_model.close()

    print("Obtain savedmodel predictions")
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

    with tf.compat.v1.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.SERVING],
            saved_model_path
        )

        predictions = sess.run(
            'intent/predictions_intent/predictions_intent:0',
            feed_dict={
                'utterance/utterance_placeholder:0': dataset.get('utterance'),
            }
        )

        savedmodel_predictions_df = pd.DataFrame(
            data=[train_set_metadata['intent']["idx2str"][p] for p in
                  predictions], columns=['intent_predictions'])
        savedmodel_predictions_df.to_csv('saved_model_predictions.csv',
                                         index=False)

        savedmodel_weights = sess.run('utterance/fc_0/weights:0')
        print(savedmodel_weights[0])

    import numpy as np
    print("Are the weights identical?",
          np.all(ludwig_weights == savedmodel_weights))
    print("Are the predictions identical?",
          np.all(ludwig_predictions_df['intent_predictions'] ==
                 savedmodel_predictions_df['intent_predictions']))


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

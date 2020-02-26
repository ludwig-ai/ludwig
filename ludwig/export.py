import argparse
import shutil
import sys

import tensorflow as tf
from tensorflow.compat.v1 import saved_model

from ludwig.contrib import contrib_command
from ludwig.models.model import load_model_and_definition


def export(
    ludwig_model_path = None,
    export_path = None,
):
    if ludwig_model_path == None:
        raise ValueError('ludwig_model_path is required')

    if export_path == None:
        raise ValueError('export_path is required')

    model, model_definition = load_model_and_definition(ludwig_model_path)
    ludwig_weights = model.collect_weights(['utterance/fc_0/weights:0'])[
        'utterance/fc_0/weights:0']
    print(ludwig_weights[0])

    print('Successfully loaded ludwig model')

    inputs = {}
    outputs = {}

    for feature in model_definition['input_features']:
        inputs[feature['name']] = getattr(model, feature['name'])
        break

    for feature in model_definition['output_features']:
        outputs[feature['name']] = getattr(model,
                                           'predictions_' + feature['name'])
        break

    print('=== Inputs ===')
    print(inputs)
    print('=== Outputs ===')
    print(outputs)

    shutil.rmtree(export_path, ignore_errors=True)
    builder = saved_model.builder.SavedModelBuilder(export_path)                                                                    

    model.initialize_session()

    with model.session as sess:
        signature = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
            inputs=inputs,
            outputs=outputs
        )

        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tf.saved_model.SERVING],
            signature_def_map={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature
            })

        builder.save()

        # savedmodel_weights = sess.run('utterance/fc_0/weights:0')
        # print(savedmodel_weights[0])

    print('Successfully exported model to', export_path)
    model.close_session()

    with tf.compat.v1.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.SERVING],
            export_path
        )
        savedmodel_weights = sess.run('utterance/fc_0/weights:0')
        print(savedmodel_weights[0])

    import numpy as np
    print("Are the weights identical?",
          np.all(ludwig_weights == savedmodel_weights))


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script exports model to tensorflow format',
        prog='ludwig export',
        usage='%(prog)s [options]'
    )

    parser.add_argument(
        '-lmp',
        '--ludwig_model_path',
        help='path of a pretrained ludwig model to export'
    )

    parser.add_argument(
        '-ep',
        '--export_path',
        help='export path for the model in tensorflow format'
    )

    args = parser.parse_args(sys_argv)
    export(**vars(args))

if __name__ == '__main__':
    contrib_command("export", *sys.argv)
    cli(sys.argv[1:])

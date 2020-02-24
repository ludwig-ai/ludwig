import argparse
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

    print('Successfully loaded ludwig model')

    inputs = {}
    outputs = {}

    for feature in model_definition['input_features']:
        inputs[feature['name']] = getattr(model, feature['name'])
        break

    for feature in model_definition['output_features']:
        outputs[feature['name']] = getattr(model, feature['name'])
        break

    print('=== Inputs ===')
    print(inputs)
    print('=== Outputs ===')
    print(outputs)

    builder = saved_model.builder.SavedModelBuilder(export_path)                                                                    

    model.initialize_session()

    with model.session as sess:
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs=inputs,
            outputs=outputs
        )
                                                                  
        builder.add_meta_graph_and_variables(                                                                                                        
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],                                                                                             
            signature_def_map={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature
            })
        
        builder.save()

    print('Successfully exported model to', export_path)

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

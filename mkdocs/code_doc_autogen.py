# -*- coding: utf-8 -*-
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''
Documentation Generator

This code is a modified and adapted version of Keras' code_doc_autogen.py
https://github.com/keras-team/keras/blob/master/docs/autogen.py
'''
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import os
import re
import sys

from ludwig.api import kfold_cross_validate
from ludwig.visualize import learning_curves, compare_performance, \
    compare_classifiers_performance_from_prob, \
    compare_classifiers_performance_from_pred, \
    compare_classifiers_performance_subset, \
    compare_classifiers_performance_changing_k, \
    compare_classifiers_multiclass_multimetric, compare_classifiers_predictions, \
    confidence_thresholding_2thresholds_2d, \
    confidence_thresholding_2thresholds_3d, confidence_thresholding, \
    confidence_thresholding_data_vs_acc, \
    confidence_thresholding_data_vs_acc_subset, binary_threshold_vs_metric, \
    roc_curves, roc_curves_from_test_statistics, calibration_1_vs_all, \
    calibration_multiclass, confusion_matrix, frequency_vs_f1

sys.path.append("../")

import ludwig
from ludwig.api import LudwigModel

EXCLUDE = {}

# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]
# PAGES = [
#     {
#         'page': 'models/sequential.md',
#         'methods': [
#             models.Sequential.compile,
#             models.Sequential.fit,
#             models.Sequential.evaluate,
#             models.Sequential.predict,
#             models.Sequential.train_on_batch,
#             models.Sequential.test_on_batch,
#             models.Sequential.predict_on_batch,
#             models.Sequential.fit_generator,
#             models.Sequential.evaluate_generator,
#             models.Sequential.predict_generator,
#             models.Sequential.get_layer,
#         ],
#         'functions': [
#             preprocessing.sequence.pad_sequences,
#             preprocessing.sequence.skipgrams,
#             preprocessing.sequence.make_sampling_table,
#         ],
#         'classes': [
#             preprocessing.sequence.TimeseriesGenerator,
#         ],
#         'all_module_functions': [initializers],
#         'all_module_classes': [initializers]
#     }
# ]

PAGES = [
    {
        'page': 'api/LudwigModel.md',
        'classes': [
            (LudwigModel, "*")
        ],
        'functions': [
            kfold_cross_validate
        ]
    },
    {
        'page': 'api/visualization.md',
        'functions': [
            learning_curves,
            compare_performance,
            compare_classifiers_performance_from_prob,
            compare_classifiers_performance_from_pred,
            compare_classifiers_performance_subset,
            compare_classifiers_performance_changing_k,
            compare_classifiers_multiclass_multimetric,
            compare_classifiers_predictions,
            confidence_thresholding_2thresholds_2d,
            confidence_thresholding_2thresholds_3d,
            confidence_thresholding,
            confidence_thresholding_data_vs_acc,
            confidence_thresholding_data_vs_acc_subset,
            binary_threshold_vs_metric,
            roc_curves,
            roc_curves_from_test_statistics,
            calibration_1_vs_all,
            calibration_multiclass,
            confusion_matrix,
            frequency_vs_f1,
        ]
    }
    # {
    #     'page': 'api/sequential_encoders.md',
    #     'classes': [
    #         (sequence_encoders.EmbedEncoder, "*"),
    #         (sequence_encoders.ParallelCNN, "*"),
    #         (sequence_encoders.StackedCNN, "*"),
    #         (sequence_encoders.StackedParallelCNN, "*"),
    #         (sequence_encoders.RNN, "*"),
    #         (sequence_encoders.CNNRNN, "*"),
    #     ]
    # }
]

ROOT = 'http://ludwig.ai/'
OUTPUT_DIR = 'docs'


def get_function_signature(_function, _method=True):
    wrapped = getattr(_function, '_original_function', None)
    if wrapped is None:
        _signature = inspect.getfullargspec(_function)
    else:
        _signature = inspect.getfullargspec(wrapped)
    defaults = _signature.defaults
    if _method and _signature.args and _signature.args[0] == 'self':
        args = _signature.args[1:]
    else:
        args = _signature.args
    if defaults:
        kwargs = zip(args[-len(defaults):], defaults)
        args = args[:-len(defaults)]
    else:
        kwargs = []
    st = '%s.%s(\n' % (
        clean_module_name(_function.__module__), _function.__name__)

    for a in args:
        st += '  {},\n'.format(str(a))
    for a, v in kwargs:
        if isinstance(v, str):
            v = '\'' + v + '\''
        st += '  {}={},\n'.format(str(a), str(v))
    if kwargs or args:
        _signature = st[:-2] + '\n)'
    else:
        _signature = st + ')'
    return post_process_signature(_signature)


def get_class_signature(_cls):
    try:
        class_signature = get_function_signature(_cls.__init__)
        class_signature = class_signature.replace('__init__', _cls.__name__)
    except (TypeError, AttributeError):
        # in case the class inherits from object and does not
        # define __init__
        class_signature = "{clean_module_name}.{cls_name}()".format(
            clean_module_name=clean_module_name(_cls.__module__),
            cls_name=_cls.__name__
        )
    return post_process_signature(class_signature)


def post_process_signature(_signature):
    parts = re.split(r'\.(?!\d)', _signature)
    if len(parts) >= 4:
        if parts[1] == 'api':
            _signature = 'ludwig.' + '.'.join(parts[2:])
    return _signature


def clean_module_name(name):
    # if name.startswith('ludwig.api'):
    #    name = name.replace('ludwig.api', 'ludwig')
    return name


def class_to_docs_link(_cls):
    module_name = clean_module_name(_cls.__module__)
    module_name = module_name[6:]
    link = ROOT + module_name.replace('.', '/') + '#' + _cls.__name__.lower()
    return link


def class_to_source_link(_cls):
    module_name = clean_module_name(_cls.__module__)
    _path = module_name.replace('.', '/')
    _path += '.py'
    line = inspect.getsourcelines(_cls)[-1]
    link = ('https://github.com/uber/'
            'ludwig/blob/master/' + _path + '#L' + str(line))
    return '[[source]](' + link + ')'


def code_snippet(snippet):
    result = '```python\n'
    result += snippet + '\n'
    result += '```\n'
    return result


def count_leading_spaces(s):
    ws = re.search(r'\S', s)
    if ws:
        return ws.start()
    else:
        return 0


def process_list_block(_docstring, starting_point, section_end,
                       leading_spaces, marker):
    ending_point = _docstring.find('\n\n', starting_point)
    block = _docstring[starting_point:(None if ending_point == -1 else
                                      ending_point - 1)]
    # Place marker for later reinjection.
    docstring_slice = _docstring[
                      starting_point:None if section_end == -1 else section_end].replace(
        block, marker)
    _docstring = (_docstring[:starting_point]
                 + docstring_slice
                 + _docstring[section_end:])
    lines = block.split('\n')
    # Remove the computed number of leading white spaces from each line.
    lines = [re.sub('^' + ' ' * leading_spaces, '', line) for line in lines]
    # Usually lines have at least 4 additional leading spaces.
    # These have to be removed, but first the list roots have to be detected.
    top_level_regex = r'^    ([^\s\\\(]+):(.*)'
    top_level_replacement = r'- __\1__:\2'
    lines = [re.sub(top_level_regex, top_level_replacement, line) for line in
             lines]
    # All the other lines get simply the 4 leading space (if present) removed
    lines = [re.sub(r'^    ', '', line) for line in lines]
    # Fix text lines after lists
    indent = 0
    text_block = False
    for i in range(len(lines)):
        line = lines[i]
        spaces = re.search(r'\S', line)
        if spaces:
            # If it is a list element
            if line[spaces.start()] == '-':
                indent = spaces.start() + 1
                if text_block:
                    text_block = False
                    lines[i] = '\n' + line
            elif spaces.start() < indent:
                text_block = True
                indent = spaces.start()
                lines[i] = '\n' + line
        else:
            text_block = False
            indent = 0

    # deal with rst
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith(':param '):
            line = line[7:]
            pos_second_colon = line.find(':')
            param_name = line[:pos_second_colon]
            pos_open_bracket = line.find('(')
            pos_close_bracket = line.find(')')
            inside_brackets = line[pos_open_bracket + 1:pos_close_bracket]
            description = line[pos_close_bracket + 1:]
            lines[i] = '- __{}__ ({}):{}'.format(param_name,
                                                 inside_brackets,
                                                 description)
        elif line.startswith(':return:'):
            line = line[8:]
            inside_brackets = ''
            pos_open_bracket = line.find('(')
            pos_close_bracket = line.rfind(')')
            if pos_open_bracket and pos_close_bracket:
                inside_brackets = ' ({})'.format(
                    line[pos_open_bracket + 1:pos_close_bracket])
                line = line[pos_close_bracket + 1:]
            lines[i] = '- __return__{}:{}'.format(inside_brackets, line)

    block = '\n'.join(lines)
    return _docstring, block


def process_docstring(_docstring):
    # First, extract code blocks and process them.
    code_blocks = []
    if '```' in _docstring:
        tmp = _docstring[:]
        while '```' in tmp:
            tmp = tmp[tmp.find('```'):]
            index = tmp[3:].find('```') + 6
            snippet = tmp[:index]
            # Place marker in docstring for later reinjection.
            _docstring = _docstring.replace(
                snippet, '$CODE_BLOCK_%d' % len(code_blocks))
            snippet_lines = snippet.split('\n')
            # Remove leading spaces.
            num_leading_spaces = snippet_lines[-1].find('`')
            snippet_lines = ([snippet_lines[0]] +
                             [line[num_leading_spaces:]
                              for line in snippet_lines[1:]])
            # Most code snippets have 3 or 4 more leading spaces
            # on inner lines, but not all. Remove them.
            inner_lines = snippet_lines[1:-1]
            leading_spaces = None
            for line in inner_lines:
                if not line or line[0] == '\n':
                    continue
                spaces = count_leading_spaces(line)
                if leading_spaces is None:
                    leading_spaces = spaces
                if spaces < leading_spaces:
                    leading_spaces = spaces
            if leading_spaces:
                snippet_lines = ([snippet_lines[0]] +
                                 [line[leading_spaces:]
                                  for line in snippet_lines[1:-1]] +
                                 [snippet_lines[-1]])
            snippet = '\n'.join(snippet_lines)
            code_blocks.append(snippet)
            tmp = tmp[index:]

    # Format docstring lists.
    section_regex = r'\n( +)# (.*)\n'
    section_idx = re.search(section_regex, _docstring)
    shift = 0
    sections = {}
    while section_idx and section_idx.group(2):
        anchor = section_idx.group(2)
        leading_spaces = len(section_idx.group(1))
        shift += section_idx.end()
        next_section_idx = re.search(section_regex, _docstring[shift:])
        if next_section_idx is None:
            section_end = -1
        else:
            section_end = shift + next_section_idx.start()
        marker = '$' + anchor.replace(' ', '_') + '$'
        _docstring, content = process_list_block(_docstring,
                                                 shift,
                                                 section_end,
                                                 leading_spaces,
                                                 marker)
        sections[marker] = content
        # `docstring` has changed, so we can't use `next_section_idx` anymore
        # we have to recompute it
        section_idx = re.search(section_regex, _docstring[shift:])

    # Format docstring section titles.
    _docstring = re.sub(r'\n(\s+)# (.*)\n',
                       r'\n\1__\2__\n\n',
                        _docstring)

    # Strip all remaining leading spaces.
    # generator function to resolve deepsource.io major issue
    def strip_leading_spaces(these_lines):
        for l in these_lines:
            yield l.lstrip(' ')

    lines = _docstring.split('\n')
    _docstring = '\n'.join(strip_leading_spaces(lines))

    # Reinject list blocks.
    for marker, content in sections.items():
        _docstring = _docstring.replace(marker, content)

    # Reinject code blocks.
    for i, code_block in enumerate(code_blocks):
        _docstring = _docstring.replace(
            '$CODE_BLOCK_%d' % i, code_block)
    return _docstring


def read_file(_path):
    with open(_path) as _f:
        return _f.read()


def collect_class_methods(_cls, _methods):
    if isinstance(_methods, (list, tuple)):
        return [getattr(_cls, m) if isinstance(m, str) else m for m in _methods]
    _methods = []
    for _, _method in inspect.getmembers(_cls, predicate=inspect.isroutine):
        if _method.__name__[0] == '_' or _method.__name__ in EXCLUDE:
            continue
        _methods.append(_method)
    return _methods


def render_function(_function, _method=True):
    _subblocks = []
    _signature = get_function_signature(_function, _method=_method)
    if _method:
        _signature = _signature.replace(
            clean_module_name(_function.__module__) + '.', '')
    _subblocks.append('## ' + _function.__name__ + '\n')
    _subblocks.append(code_snippet(_signature))
    _docstring = _function.__doc__
    if _docstring:
        _subblocks.append(process_docstring(_docstring))
    return '\n\n'.join(_subblocks)


def read_page_data(_page_data, type):
    assert type in ['classes', 'functions', 'methods']
    data = _page_data.get(type, [])
    for module in _page_data.get('all_module_{}'.format(type), []):
        module_data = []
        for name in dir(module):
            if name[0] == '_' or name in EXCLUDE:
                continue
            module_member = getattr(module, name)
            if (inspect.isclass(module_member) and type == 'classes' or
                    inspect.isfunction(module_member) and type == 'functions'):
                instance = module_member
                if module.__name__ in instance.__module__:
                    if instance not in module_data:
                        module_data.append(instance)
        module_data.sort(key=lambda x: id(x))
        data += module_data
    return data


if __name__ == '__main__':
    print('Cleaning up existing {} directory.'.format(OUTPUT_DIR))
    for page_data in PAGES:
        file_path = os.path.join(OUTPUT_DIR, page_data['page'])
        if os.path.exists(file_path):
            os.remove(file_path)
    # if os.path.exists(OUTPUT_DIR):
    #   shutil.rmtree(OUTPUT_DIR)

    # print('Populating {} directory with templates.'.format(OUTPUT_DIR))
    # for subdir, dirs, fnames in os.walk('templates'):
    #    for fname in fnames:
    #       new_subdir = subdir.replace('templates', OUTPUT_DIR)
    #        if not os.path.exists(new_subdir):
    #            os.makedirs(new_subdir)
    #        if fname[-3:] == '.md':
    #            fpath = os.path.join(subdir, fname)
    #            new_fpath = fpath.replace('templates', OUTPUT_DIR)
    #            shutil.copy(fpath, new_fpath)

    # readme = read_file('../README.md.md')
    # index = read_file('templates/index.md')
    # index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
    # with open(os.path.join(OUTPUT_DIR, 'index.md'), 'w') as f:
    #    f.write(index)

    print('Generating docs for Ludwig %s.' % ludwig.__version__)
    for page_data in PAGES:
        classes = []
        functions = []
        methods = []
        if 'classes' in page_data.keys():
            classes = read_page_data(page_data, 'classes')
        elif 'functions' in page_data.keys():
            functions = read_page_data(page_data, 'functions')
        else:
            raise TypeError('Invalid type specified in page_data')

        blocks = []
        for element in classes:
            if not isinstance(element, (list, tuple)):
                element = (element, None)
            cls = element[0]
            subblocks = []
            signature = get_class_signature(cls)
            subblocks.append('<span style="float:right;">' +
                             class_to_source_link(cls) + '</span>')
            if element[1]:
                subblocks.append('# ' + cls.__name__ + ' class\n')
            else:
                subblocks.append('## ' + cls.__name__ + '\n')
            subblocks.append(code_snippet(signature))
            docstring = cls.__doc__
            if docstring:
                subblocks.append(process_docstring(docstring))
            methods = collect_class_methods(cls, element[1])
            if methods:
                subblocks.append('\n---')
                subblocks.append('# ' + cls.__name__ + ' methods\n')

                # generator function to resolve deepsource.io major issue
                def generate_render_functions(_methods):
                    for m in _methods:
                        yield render_function(m, _method=True)

                subblocks.append('\n---\n'.join(generate_render_functions(methods)))
            blocks.append('\n'.join(subblocks))

        methods = read_page_data(page_data, 'methods')

        for method in methods:
            blocks.append(render_function(method, _method=True))

        functions = read_page_data(page_data, 'functions')

        if functions:
            blocks.append('# Module functions\n')

        for function in functions:
            blocks.append(render_function(function, _method=False))

        if not blocks:
            raise RuntimeError('Found no content for page ' +
                               page_data['page'])

        mkdown = '\n----\n\n'.join(blocks)
        # save module page.
        # Either insert content into existing page,
        # or create page otherwise
        page_name = page_data['page']
        path = os.path.join(OUTPUT_DIR, page_name)
        if os.path.exists(path):
            template = read_file(path)
            assert '{{autogenerated}}' in template, (
                    'Template found for ' + path +
                    ' but missing {{autogenerated}}'
                    ' tag.')
            mkdown = template.replace('{{autogenerated}}', mkdown)
            print('...inserting autogenerated content into template:', path)
        else:
            print('...creating new page with autogenerated content:', path)
        subdir = os.path.dirname(path)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        with open(path, 'w') as f:
            f.write(mkdown)

    # shutil.copyfile('../CONTRIBUTING.md', 'os.path.join(OUTPUT_DIR, 'contributing.md'))

# coding=utf-8
'''Ludwig: a deep learning experimentation toolbox
'''
from codecs import open
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README.md file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ludwig',

    version='0.1.0',

    description='A deep learning experimentation toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://ludwig.ai',

    author='Piero Molino',
    author_email='piero.molino@gmail.com',

    license='Apache 2.0',

    keywords='ludwig deep learning deep_learning machine machine_learning natural language processing computer vision',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    python_requires='>=3',

    include_package_data=True,
    package_data={'ludwig': ['etc/*', 'examples/*.py']},

    install_requires=['Cython>=0.25',
                      'h5py>=2.6',
                      'matplotlib>=2.0',
                      'numpy>=1.12',
                      'pandas>=0.19',
                      'scipy>=0.18',
                      'scikit-learn',
                      'scikit-image',
                      'seaborn>=0.7',
                      'spacy>=2.0',
                      'tqdm',
                      'tabulate>=0.7',
                      'tensorflow>=1.12',
                      'PyYAML>=3.12'
                      ],

    entry_points={
        'console_scripts': [
            'ludwig=ludwig.cli:main'
        ]
    }
)

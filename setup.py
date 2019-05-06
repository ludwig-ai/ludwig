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

def read(fname):
    with open(fname, 'r') as fhandle:
        return fhandle.read()

def read_reqs(fname):
    req_path = path.join(here, fname)
    return [req.strip() for req in read(req_path).splitlines() if req.strip()]


gcs_reqs = read_reqs('requirements-gcs.txt')
all_reqs = gcs_reqs
extras_require = {
    "all": all_reqs,
    "gcs": gcs_reqs,
 }

setup(
    name='ludwig',
    version='0.1.2',
    description='A deep learning experimentation toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://ludwig.ai',
    author='Piero Molino',
    author_email='piero.molino@gmail.com',
    license='Apache 2.0',
    keywords='ludwig deep learning deep_learning machine machine_learning '
    'natural language processing computer vision',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    python_requires='>=3',
    include_package_data=True,
    package_data={'ludwig': ['etc/*', 'examples/*.py']},
    install_requires=read_reqs('requirements.txt'),
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'ludwig=ludwig.cli:main'
        ]
    }
)

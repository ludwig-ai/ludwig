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

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line]

extra_requirements = {}

with open(path.join(here, 'requirements_audio.txt'), encoding='utf-8') as f:
    extra_requirements['audio'] = [line.strip() for line in f if line]

with open(path.join(here, 'requirements_image.txt'), encoding='utf-8') as f:
    extra_requirements['image'] = [line.strip() for line in f if line]

with open(path.join(here, 'requirements_serve.txt'), encoding='utf-8') as f:
    extra_requirements['serve'] = [line.strip() for line in f if line]

with open(path.join(here, 'requirements_text.txt'), encoding='utf-8') as f:
    extra_requirements['text'] = [line.strip() for line in f if line]

with open(path.join(here, 'requirements_viz.txt'), encoding='utf-8') as f:
    extra_requirements['viz'] = [line.strip() for line in f if line]

with open(path.join(here, 'requirements_horovod.txt'), encoding='utf-8') as f:
    extra_requirements['horovod'] = [line.strip() for line in f if line]

with open(path.join(here, 'requirements_dask.txt'), encoding='utf-8') as f:
    extra_requirements['dask'] = [line.strip() for line in f if line]

with open(path.join(here, 'requirements_ray.txt'), encoding='utf-8') as f:
    extra_requirements['ray'] = [line.strip() for line in f if line] \
                                + extra_requirements['dask'] \
                                + extra_requirements['horovod']

with open(path.join(here, 'requirements_hyperopt.txt'), encoding='utf-8') as f:
    extra_requirements['hyperopt'] = [line.strip() for line in f if line]

extra_requirements['full'] = [item for sublist in extra_requirements.values()
                              for item in sublist]

with open(path.join(here, 'requirements_test.txt'), encoding='utf-8') as f:
    extra_requirements['test'] = (
            extra_requirements['full'] + [line.strip() for line in f if line]
    )

setup(
    name='ludwig',
    version='0.4-dev0',

    description='A deep learning experimentation toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/ludwig-ai/ludwig',
    download_url='https://pypi.org/project/ludwig/',

    author='Piero Molino',
    author_email='piero.molino@gmail.com',

    license='Apache 2.0',

    keywords='ludwig deep learning deep_learning machine machine_learning natural language processing computer vision',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    python_requires='>=3.6',

    include_package_data=True,
    package_data={'ludwig': ['etc/*', 'examples/*.py']},

    install_requires=requirements,
    extras_require=extra_requirements,

    entry_points={
        'console_scripts': [
            'ludwig=ludwig.cli:main'
        ]
    }
)

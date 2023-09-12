#!/usr/bin/env python

from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "SaGui is designed to work with Python 3.6 and greater. " \
    + "Please install it before proceeding."

setup(
    name='sagui',
    packages=['sagui'],
    install_requires=[
        # 'cython<3',
        # 'protobuf==3.20.0',
        # 'numpy~=1.17.4',
        # 'gym~=0.15.3',
        # 'joblib==0.14.0',
        # 'matplotlib==3.1.1',
        # 'mujoco_py==2.0.2.5',
        # 'seaborn==0.8.1',
        # 'tensorflow==1.13.1',

        # From https://github.com/openai/spinningup/blob/master/setup.py
        # 'cloudpickle==1.2.1',
        # 'gym[atari,box2d,classic_control]~=0.15.3',
        # 'ipython',
        # 'joblib',
        # 'matplotlib==3.1.1',
        # 'mpi4py',
        # 'numpy',
        # 'pandas',
        # 'pytest',
        # 'psutil',
        # 'scipy',
        # 'seaborn==0.8.1',
        # 'tensorflow>=1.8.0,<2.0',
        # 'torch==1.3.1',
        # 'tqdm',
        # 'mpi4py',
    ],
)

#!/usr/bin/env python

import setuptools

setuptools.setup(
    name='blocksparse',
    version='1.13.1',
    description='Tensorflow ops for blocksparse matmul, transformer, convolution and related operations.',
    author='OpenAI',
    maintainer='Scott Gray',
    maintainer_email='scott@openai.com',
    install_requires=[
        'numpy',
        'scipy',
        # We don't depend on `tensorflow` or `tensorflow-gpu` here, since one or the other is sufficient.
    ],
    packages=['blocksparse'],
    package_data={ 'blocksparse': ['blocksparse_ops.so'] },
    url='https://github.com/openai/blocksparse',
    license='MIT')

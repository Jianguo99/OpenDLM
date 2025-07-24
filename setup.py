# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

from setuptools import find_packages
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'opendlm/VERSION')) as f:
    version = f.read().strip()

setup(name='opendlm',
      version=version,
      url='https://github.com/Jianguo99/OpenDLM',
      package_data={'examples': ['*.ipynb']},
      description="A Python toolbox for diffusion language models focusing on sampling strategies.",
      install_requires=[],
      include_package_data=True,
      packages=find_packages(),
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      )

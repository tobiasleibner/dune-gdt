#!/usr/bin/env python
#
# This file is part of the dune-gdt project:
#   https://github.com/dune-community/dune-gdt
# Copyright 2010-2017 dune-gdt developers and contributors. All rights reserved.
# License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
#          with "runtime exception" (http://www.dune-project.org/license.html)
# Authors:
#   Felix Schindler (2017)
#   Rene Milk       (2016)

import sys
from setuptools import setup

setup(name='dune.gdt',
      version='0.3-dev',
      namespace_packages=['dune'],
      description='Python for Dune-Gdt',
      author='The dune-gdt devs',
      author_email='dune-gdt-dev@listserv.uni-muenster.de',
      url='https://github.com/dune-community/dune-gdt',
      packages=['dune.gdt'],
      install_requires=['jinja2', 'where'],
      )

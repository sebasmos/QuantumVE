#!/usr/bin/env python

from distutils.core import setup

setup(
    name='fokker_planck',
    version='1.0.0',
    url='https://github.com/juliankappler/',
    author='Julian Kappler',
    author_email='jkappler@posteo.de',
    license='GPL3',
    description='python module with numerical and analytical tools for the 1D' \
                + ' fokker planck equation',
    long_description='fokker_planck is a python module which contains ' \
                     + 'both numerical and analytical tools for the 1D ' \
                     + 'Fokker-Planck equation.',
    packages=['fokker_planck'],
)

# encoding: utf-8
from setuptools import setup

setup(name='vgs',
      version='0.2',
      description='Visually grounded speech models in Pytorch',
      url='https://github.com/gchrupala/vgs',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      packages=['onion','vg'],
      zip_safe=False,
      install_requires=[
          'torch',
          'torchvision',
          'sklearn',
          'scipy'
      ])

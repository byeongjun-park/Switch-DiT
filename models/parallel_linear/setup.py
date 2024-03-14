from setuptools import setup, find_packages

setup(name='parallel_experts',
      packages=find_packages(), 
      install_requires=[
            'torch'
      ])
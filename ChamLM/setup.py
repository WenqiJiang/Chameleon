from setuptools import setup

setup(name='ralm',
      version='0.0.1',
      author='Wenqi Jiang and Marco Zeller',
      packages=['ralm'],
      description='A PyTorch re-implementation of Retrieval-Augmented Language Model',
      license='MIT',
      install_requires=[
            'torch',
      ],
)
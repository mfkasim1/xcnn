import os
from setuptools import setup, find_packages

module_name = "xcdnn2"

setup(
    name='xcdnn2',
    version="0.1.0",
    description='Retrieving XC with DNN',
    url='https://github.com/mfkasim1/xcdnn2',
    author='mfkasim1',
    author_email='firman.kasim@gmail.com',
    license='MIT',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.8.2",
        "pyyaml>=5.3.1",
        # "pytorch>=1.8.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",

        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="project library deep-learning dft",
    zip_safe=False
)

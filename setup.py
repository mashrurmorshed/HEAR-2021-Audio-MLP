import setuptools
from setuptools import setup

setup(
    name="audiomlp",
    version="2.0.0",
    description="MLP-based feature encoder for audio.",
    url="https://github.com/ID56/HEAR-2021-Audio-MAE",
    author="Mashrur M. Morshed",
    author_email="mashrurmahmud@iut-dhaka.edu",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "einops",
        "nnAudio"   
    ]
)
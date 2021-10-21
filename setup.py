from setuptools import setup

setup(
    name="audiomlp",
    version="0.0.1",
    description="MLP-based feature encoder for audio.",
    url="https://github.com/ID56/Audio-MLP",
    author="Mashrur M. Morshed",
    author_email="mashrurmahmud@iut-dhaka.edu",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "librosa",
        "einops"   
    ]
)
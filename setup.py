from setuptools import setup
from setuptools import find_packages

setup(
    name="sae",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch"],
    author="Ryan Kortvelesy",
    author_email="rk627@cam.ac.uk",
    description="A Set Autoencoder. Functions as a neural network which can produce a variable number of outputs.",
)

# setup.py

from setuptools import setup, find_packages

setup(
    name="secret_scrimmage_lineup_guesser",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "pulp",
        "tqdm",
    ],
    author="Robert Robison",
    author_email="rob.a.rrobison@gmail.com",
    description="A package to guess lineup for secret scrimmage games.",
    url="https://github.com/robert.robison/secret_scrimmage-lineup-guesser",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
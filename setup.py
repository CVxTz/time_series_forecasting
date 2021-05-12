import os

from setuptools import find_packages
from setuptools import setup

path = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except FileNotFoundError:
    REQUIRED = []

setup(
    name="time_series_forecasting",
    version="0.1",
    description="Time series forecasting",
    author="TimeSeries",
    url="https://github.com/CVxTz/time_series_forecasting",
    install_requires=REQUIRED,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(exclude=("example", "app", "data", "docker", "tests")),
)

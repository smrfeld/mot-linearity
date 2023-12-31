from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="motlinearity",
    version="0.1.0",
    author="Oliver K. Ernst",
    description="Analyze linear statistics of MOT datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smrfeld/mot-linearity",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6"
)
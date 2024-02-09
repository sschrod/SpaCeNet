from setuptools import setup, find_packages

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
  name = "SpaCeNet",
  version = "0.0.1",
  author = "Niklas Lueck",
  author_email = "niklas.lueck@stud.uni-goettingen.de",
  description = "Implements SpaCeNet with (group) LASSO regularization.",
  long_description = long_description,
  long_description_content_type = "text/markdown",
  packages = find_packages(),
  py_modules = [
    "optimizer",
    "model",
    "utils"
  ],
  install_requires = [
    "matplotlib>=3.4.3",
    "numpy>=1.21",
    "pandas>=1.3.3",
    "torch>=1.10.1",
    "tqdm>=4.62.3"
  ]
)

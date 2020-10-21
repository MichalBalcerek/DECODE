import os
import setuptools

from setuptools import setup

# requirements when building the wheel via pip; conda install uses
#   info in meta.yaml instead; we're supporting multiple environments, thus
#   we have to accept some duplication (or near-duplication), unfortunately;
#   however, if conda sees the requirements here, it will be unhappy
if "CONDA_BUILD" in os.environ:
    # conda requirements set in meta.yaml
    requirements = []
else:
    # pip needs requirements here; keep in sync with meta.yaml!
    requirements = [
        "numpy",
        "torch",
        "torchvision",
        "click",
        "deprecated",
        "h5py",
        "matplotlib",
        "pandas",
        "pytest",
        "pyyaml",
        "requests",
        "scipy",
        "seaborn",
        "scikit-image",
        "scikit-learn",
        "tensorboard",
        "tifffile",
        "tqdm",
        ]

setup(
    name='decode',
    version='0.9.3.a0',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    url='https://rieslab.de',
    license='GPL3',
    author='Lucas-Raphael Mueller',
    author_email='',
    description=''
)

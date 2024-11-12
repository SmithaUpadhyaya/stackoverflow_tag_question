from setuptools import find_packages, setup

__version__ = "2.1.2"

REPO_NAME = "StackOverflow Tag Questions"
AUTHOR_USER_NAME = "SU"
LICENSE = "MIT"


setup(
    name = REPO_NAME,
    version = __version__,
    author = AUTHOR_USER_NAME,
    license = LICENSE,
    packages = find_packages(),    
    
)

#Command
#Init DVC: dvc init
#Destroy DVC: dvc destroy
from typing import List
from setuptools import find_packages,setup


HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This funcction will return the List of requirements
    '''
    requirements=[]
    with open(file_path) as file_object:
        requirements=file_object.readlines()
        requirements=[req.replace("\n","") for req in requirements]  #As while reading from requirements.txt it will pickup line changes i.e. "\n"
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)                           #As while reading from requirements.txt it will pickup -e . which is basically used to call setup.py automatically
    return requirements


setup(
    name = 'MLProject',
    version='0.0.1',
    author = 'Divyam',
    author_email='divyamm298@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
#find_packages will go inside all the folers and find __init__.py file and build all the files in the folder as a package so that we can import it anywhere

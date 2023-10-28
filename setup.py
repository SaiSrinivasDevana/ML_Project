from setuptools import find_packages,setup
from typing import List
Char="-e ."

def get_requirements(filepath :str)->List[str]:
    requirements=[]
    with open(filepath) as fileobj:
        requirements=fileobj.readlines()
        requirements=[r.replace('\n','')for r in requirements]

        if Char in requirements:
            requirements.remove(Char)
    return requirements

setup(
    name="MLProject",
    version='0.0.1',
    author='SaiSrinivasDevana',
    author_email='saidevana12@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)
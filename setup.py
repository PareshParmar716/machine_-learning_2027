
from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the list of requirements.
    """

    with open(file_path, "r") as file_obj:
        requirements = [
            req.strip()
            for req in file_obj.readlines()
            if req.strip() and not req.startswith("#")
        ]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="mlops_project",
    version="0.0.1",
    author="Parmar Paresh",
    author_email="pp6003271@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)

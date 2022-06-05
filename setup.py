from setuptools import setup


setup(
    name="chameleon",
    version="1.0.0",
    author="Adrian Brodzik",
    description=("Chameleon clustering algorithm."),
    license="MIT",
    keywords="chameleon clustering",
    packages=["chameleon"],
    install_requires=[i.replace("\n", "") for i in open("requirements.txt", "r").readlines()],
)

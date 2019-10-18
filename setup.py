from setuptools import setup, find_packages


with open('A:/Capstone_Code/manda_prediction/requirements.txt') as requirements:
    required = requirements.read().splitlines()

setup(
    name='mandaprediction',
    version='0.1.0',
    description='manda',
    #url='https://github.com/taspinar/twitterscraper',
    author=['', ''],
    packages=find_packages(exclude=["build.*", "tests", "tests.*"]),
    install_requires=required,
    entry_points={
        "console_scripts": [
            "MandA = MandA.main:main"
        ]
    })

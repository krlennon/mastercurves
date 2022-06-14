from setuptools import setup, find_packages
import codecs
import os

def read(*parts):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()

VERSION = '0.2.2'
DESCRIPTION = 'Python package for building master curves from data'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="mastercurves",
        version=VERSION,
        author="Kyle Lennon",
        author_email="<kyle.lennon08@gmail.com>",
        description=DESCRIPTION,
        license="GNU GPLv3",
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        url="https://github.com/krlennon/mastercurves",
        packages=find_packages(),
        install_requires=["numpy", "matplotlib", "scikit-learn", "scipy", "pandas", "numdifftools"],
        keywords=['python', 'master', 'curves', 'mastercurves', 'Bayesian', 'Gaussian', 'process', 'regression', 'machine', 'learning', 'statistics'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: End Users/Desktop",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)

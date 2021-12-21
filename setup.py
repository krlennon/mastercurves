from setuptools import setup, find_packages

VERSION = '0.1.2'
DESCRIPTION = 'Python package for building master curves from data'
LONG_DESCRIPTION = 'Python package for building master curves from data. Uses Gaussian process regression and Bayesian inference to build statically robust master curves from parametrically self-similar data, with uncertainty estimates.'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="mastercurves",
        version=VERSION,
        author="Kyle Lennon",
        author_email="<kyle.lennon08@gmail.com>",
        description=DESCRIPTION,
        license="GNU GPLv3",
        long_description=LONG_DESCRIPTION,
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

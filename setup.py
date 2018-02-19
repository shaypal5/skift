"""Setup for the skift package."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
import versioneer


# DEPENDENCY_LINKS = [
#     (
#         'git+https://github.com/facebookresearch/fastText.git'
#         '@3b5fd293597de550a131d81436b9755902f39bb2'
#         '#egg=fasttext-0.1.0+git.3b5fd29'
#     ),
# ]
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'scikit-learn',
]
# FT_REQUIRES = INSTALL_REQUIRES + [
#     'fasttext==0.1.0+git.3b5fd29',
# ]
TEST_REQUIRES = [
    # testing and coverage
    'pytest', 'coverage', 'pytest-cov',
    # unmandatory dependencies of the package itself
    'pandas',
    # to be able to run `python setup.py checkdocs`
    'collective.checkdocs', 'pygments',
]

with open('README.rst') as f:
    README = f.read()

setuptools.setup(
    author="Shay Palachy",
    author_email="shay.palachy@gmail.com",
    name='skift',
    description='scikit-learn wrappers for Python fastText',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description=README,
    url='https://github.com/shaypal5/skift',
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.5",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES + INSTALL_REQUIRES,
        # 'fasttext': FT_REQUIRES,
    },
    # dependency_links=DEPENDENCY_LINKS,
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

setup(
    name='urbansim',
    version='0.2dev',
    description='Tool for modeling metropolitan real estate markets',
    author='Synthicity',
    author_email='ffoti@berkeley.edu',
    license='AGPL',
    url='https://github.com/synthicity/urbansim',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: GNU Affero General Public License v3'
    ],
    package_data = {
        '': ['*.html'],
    },
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        'bottle>=0.12.5',
        'matplotlib>=1.3.1',
        'numpy>=1.8.0',
        'pandas>=0.13.1',
        'patsy>=0.2.1',
        'prettytable>=0.7.2',
        'pyyaml>=3.10',
        'scipy>=0.13.3',
        'simplejson>=3.3.3',
        'statsmodels>=0.5.0',
        'tables>=3.1.0',
        'toolz>=0.7.0'
    ]
)
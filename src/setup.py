#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

from setuptools import setup, find_packages

setup(
    name='prediag',
    version='1.0.1',
    description='Noninvasive Prenatal Diagnosis',
    author='Ghislain Durif',
    author_email='gd.dev@libertymail.net',
    url='https://github.com/gdurif/nipd',
    license='GPL3',
    license_files = ('LICENSE.txt', 'COPYING.txt'),
    packages=find_packages(),
    install_requires=[
        "joblib",
        "numpy",
        "pandas",
        "psutil",
        "pyvcf",
        "scipy",
        "tqdm"
    ],
    entry_points={
        'console_scripts':
            ['prediag_fetal_fraction=prediag.cli.fetal_fraction_estimation:main',
             'prediag_fetal_genotyping=prediag.cli.fetal_genotyping:main',
             'prediag_fetal_allele_origin_heuristic=prediag.cli.fetal_allele_origin_heuristic:main',
             'prediag_fetal_allele_origin_bayesian=prediag.cli.fetal_allele_origin_bayesian:main'],
    },
    zip_safe=False
)

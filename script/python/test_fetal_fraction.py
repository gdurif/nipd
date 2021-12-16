#!/usr/bin/env python
## test fetal fraction

# external
import matplotlib.pyplot as plt
import numpy as np
import os
# internal
from prediag.fetal_fraction import estimate_global_fetal_fraction, impute_fetal_fraction, smooth_fetal_fraction
from prediag.filter import loci_tab_filter
from prediag.utils import float2string
from prediag.vcf_reader import load_vcf_data

## VCF files
mother_vcf = ""
father_vcf = ""
cfdna_vcf = ""

## read data
seq_data_tab = load_vcf_data(mother_vcf, father_vcf, cfdna_vcf,
                             min_rel_depth = 0.02, min_abs_depth = 2,
                             verbose = True)

## read data
seq_data_tab = loci_tab_filter(seq_data_tab, min_coverage = 50, verbose = True)

print("Data table")
print(seq_data_tab.to_string())

## fetal fraction estimation
fetal_fraction_tab = estimate_global_fetal_fraction(
    seq_data_tab, min_coverage = 60, tol = 0.05
)

print("Fetal fraction table")
print(fetal_fraction_tab.to_string(float_format = float2string))

fetal_fraction_tab_original = fetal_fraction_tab.copy()

print("fetal fraction missing values inference")
fetal_fraction_tab = impute_fetal_fraction(fetal_fraction_tab)
print(fetal_fraction_tab.to_string(float_format = float2string))

fetal_fraction_tab_imputed = fetal_fraction_tab.copy()

print("fetal fraction smoothing")
fetal_fraction_tab = smooth_fetal_fraction(fetal_fraction_tab)
print(fetal_fraction_tab.to_string(float_format = float2string))


try:
    print("Check fetal fraction missing values inference and smoothing")
    import pylab as plt
    fetal_fraction_tab['cfdna_ff_original'] = fetal_fraction_tab_original['cfdna_ff']
    fetal_fraction_tab['cfdna_ff_imputed'] = fetal_fraction_tab_imputed['cfdna_ff']
    plt.scatter(
        'pos', 'cfdna_ff_original', data=fetal_fraction_tab,
    )
    plt.plot(
        'pos', 'cfdna_ff_imputed',
        data=fetal_fraction_tab[np.isnan(fetal_fraction_tab['cfdna_ff_original'])],
        color='b', linestyle='--', marker='o'
    )
    plt.plot(
        'pos', 'cfdna_ff',
        data=fetal_fraction_tab,
        color='r', linestyle='--', marker='o'
    )
    plt.show()
except BaseException:
    print("Package 'matplotlib' not available for graphical outputs")

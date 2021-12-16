# external
import matplotlib.pyplot as plt
import numpy as np
import os
# internal
import prediag.bayesian_fetal_allele_origin as bayesian
import prediag.heuristic_fetal_allele_origin as heuristic
from prediag.fetal_fraction import estimate_global_fetal_fraction
from prediag.fetal_genotype import infer_global_fetal_genotype
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

print("Data table")
print(seq_data_tab.to_string())

## fetal fraction estimation
fetal_fraction_tab = estimate_global_fetal_fraction(
    seq_data_tab, min_coverage = 50, tol = 0.05
)

print("Fetal fraction table")
print(fetal_fraction_tab.to_string(float_format = float2string))

## fetal genotype
fetal_genotype_tab = infer_global_fetal_genotype(
                    seq_data_tab, fetal_fraction_tab.dropna(),
                    min_coverage = 50, tol = 0.0001,
                    snp_neighborhood = 50e3, n_neighbor_snp = 10,
                    return_log = False, verbose = False)

print("Fetal genotype table")
print(fetal_genotype_tab.to_string(
    float_format = float2string,
    formatters = {'fetal_gt_posterior': float2string}
))

## parental orginal haplotype init
init_allele_origin_tab = heuristic.infer_parental_allele_origin(
    fetal_genotype_tab, recombination_rate = 1.2e-8,
    genetic_dist_threshold = 1e-2, verbose = False
)

print("Parental allele origin (init)")
print(init_allele_origin_tab.to_string(
    float_format = float2string,
    formatters = {'allele_origin_conf': float2string,
                  'fetal_gt_posterior': float2string}
))

## bayesian allele origin inference
n_gibbs = 4
lag = 100
n_sample = 2000
n_iter = int(n_sample * lag / n_gibbs)
allele_origin_tab = bayesian.infer_parental_allele_origin(
    seq_data_tab, init_allele_origin_tab,
    n_burn = 2000, n_iter = n_iter, lag = lag,
    n_thread = n_gibbs, n_gibbs = n_gibbs,
    recombination_rate = 1.2e-8,
    both_parent_phased = True,
    verbose = True, filename = None
)

print("Parental allele origin")
print(allele_origin_tab.to_string(
    float_format = float2string
))

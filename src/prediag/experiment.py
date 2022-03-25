#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
import pandas as pds
import numpy as np

# internal
from prediag.simulation import multi_snp_data
import prediag.bayesian_fetal_allele_origin as bayesian
import prediag.heuristic_fetal_allele_origin as heuristic
from prediag.fetal_fraction import estimate_global_fetal_fraction
from prediag.fetal_genotype import infer_global_fetal_genotype
from prediag.utils import float2string, parse_allele_origin, index_allele_origin



def simulation(
    seq_length = 0.25, snp_dist = 2e-3, fetal_fraction = 0.2,
    recombination_rate = 1.2e-8, coverage = 200, verbose = False
):
    """Data simulation
    
    Input:
        seq_length (float): length of simulated sequence (in Mbp).
        snp_dist (float): average inter-SNP distance (in Mbp). Default is 
            `0.002 Mb`, i.e. `2 kb`.
        fetal_fraction (float): fetal fraction.
        recombination_rate (float): recombination rate in cM/Mbp 
            Default value 1.2. If None, no recombination is simulated.
        coverage (int): sequencing coverage (i.e. average number of
            single read copies). If list, variable coverage.
        verbose (bool): verbosity. Default is False.
    
    Output: see `prediag.simulation.multi_snp_data` function.
    
    """
    # additional parameters
    phased = True
    ff_constant = False
    coverage_constant = False
    add_noise = True
    # simulate
    simu_data = multi_snp_data(
        seq_length, snp_dist, phased, fetal_fraction, ff_constant, 
        recombination_rate, coverage, coverage_constant, add_noise, 
        verbose
    )
    # output
    return simu_data



def analysis_pipeline(
    simu_data, n_sample = 2000, n_burn = 100, lag = 100, n_thread = 1, 
    recombination_rate = 1.2e-8, verbose = False, **kwargs
):
    """Run full analysis pipeline with fetal fraction estimation, fetal
    genotype inference (used for Gibbs sampler initialization) and
    fetal allele origin inference (with Gibbs sampler).
    
    Input:
        simu_data (Pandas.DataFrame): sequencing data table produced
            by 'prediag.vcf_reader.load_vcf_data' function.
        n_sample (int): number of sample to sample.
        n_burn (int): number of burning iterations.
        lag (int): actually keep a sample every `lag` iterations to avoid 
            auto-correlations.
        n_thread (int): number of threads for parallel computing, if 0 then
            all cpu cores are used.
        recombination_rate (float): recombination rate per bp (between 0 and 1).
            Default value 1.2e-8.
        verbose (bool): verbosity. Default is False.
    
    Output: see `prediag.bayesian_fetal_allele_origin.infer_parental_allele_origin`
        function.
    """
    
    if verbose:
        print("Data table")
        print(simu_data.to_string())

    ## fetal fraction estimation
    fetal_fraction_tab = estimate_global_fetal_fraction(
        simu_data, min_coverage = 50, tol = 0.05
    )
    
    if verbose:
        print("Fetal fraction table")
        print(fetal_fraction_tab.to_string(float_format = float2string))

    ## fetal genotype
    fetal_genotype_tab = infer_global_fetal_genotype(
        simu_data, fetal_fraction_tab.dropna(),
        min_coverage = 50, tol = 0.0001,
        snp_neighborhood = 50e3, n_neighbor_snp = 10,
        return_log = False, verbose = False
    )
    
    if verbose:
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
    
    if verbose:
        print("Parental allele origin (init)")
        print(init_allele_origin_tab.to_string(
            float_format = float2string,
            formatters = {'allele_origin_conf': float2string,
                          'fetal_gt_posterior': float2string}
        ))

    ## bayesian allele origin inference
    n_gibbs = n_thread
    n_iter = int(n_sample * lag / n_gibbs)
    allele_origin_tab = bayesian.infer_parental_allele_origin(
        simu_data, init_allele_origin_tab,
        n_burn = n_burn, n_iter = n_iter, lag = lag,
        n_thread = n_thread, n_gibbs = n_gibbs,
        recombination_rate = recombination_rate,
        both_parent_phased = True,
        verbose = False, filename = None
    )
    
    if verbose:
        print("Parental allele origin")
        print(allele_origin_tab.to_string(
            float_format = float2string
        ))
    
    # keep track of ground truth
    allele_origin_tab = pds.merge(
        allele_origin_tab,
        simu_data[['chrom', 'pos', 'true_allele_origin']],
        how='left', on=['chrom', 'pos']
    )
    allele_origin_tab['mat_allele_origin'] = \
        allele_origin_tab['true_allele_origin'].apply(
            lambda x: parse_allele_origin(index_allele_origin(x))[0] + 1
        )
    allele_origin_tab['pat_allele_origin'] = \
        allele_origin_tab['true_allele_origin'].apply(
            lambda x: parse_allele_origin(index_allele_origin(x))[1] + 1
        )
    allele_origin_tab.drop(['true_allele_origin'], inplace=True, axis=1)
    
    # output
    return allele_origin_tab


def collect_result(allele_origin_tab):
    """Collect and summarize results for further analysis"""
    
    out = pds.DataFrame({
    
        "n_snp": [len(allele_origin_tab)], 
        "estimated_ff_av": [allele_origin_tab.fetal_fraction.mean()],
        "estimated_ff_sd": [allele_origin_tab.fetal_fraction.std()],
        
        # mother
        "mat_indecise": [np.sum(np.logical_and(
            allele_origin_tab.mat1_hap_post < 0.8, 
            allele_origin_tab.mat1_hap_post > 0.2
        ))],
        "mat_pred1": [np.sum(allele_origin_tab.mat1_hap_post >= 0.8)],
        "mat_true1": [np.sum(allele_origin_tab.mat_allele_origin == 1)],
        "mat_pred2": [np.sum(allele_origin_tab.mat2_hap_post >= 0.8)],
        "mat_true2": [np.sum(allele_origin_tab.mat_allele_origin == 2)],

        # father
        "pat_indecise": [np.sum(np.logical_and(
            allele_origin_tab.pat1_hap_post < 0.8, 
            allele_origin_tab.pat1_hap_post > 0.2
        ))],
        "pat_pred1": [np.sum(allele_origin_tab.pat1_hap_post >= 0.8)],
        "pat_true1": [np.sum(allele_origin_tab.pat_allele_origin == 1)],
        "pat_pred2": [np.sum(allele_origin_tab.pat2_hap_post >= 0.8)],
        "pat_true2": [np.sum(allele_origin_tab.pat_allele_origin == 2)],
    })
    
    # output
    return out
    
    


# example
if __name__ == '__main__':
    
    # simulate data
    simu_data = simulation(
        seq_length = 0.1, snp_dist = 2e-3, fetal_fraction = 0.2,
        recombination_rate = 1.2e-8, coverage = 200, verbose = False
    )
    
    # analysis
    res_tab = analysis_pipeline(
        simu_data, n_sample = 100, n_burn = 100, lag = 10, n_thread = 1, 
        recombination_rate = 1.2e-8, verbose = False
    )
    print(res_tab.to_string())
    
    # result
    res = collect_result(res_tab)
    print(res.to_string())

    
    

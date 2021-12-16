#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
import numpy as np
import pandas as pds
# internal
from prediag.fetal_fraction import average_fetal_fraction
from prediag.filter import loci_tab_filter, snp_check
from prediag.snp_model import fetal_genotype_prior, read_data_loglikelihood, model_posterior
from prediag.utils import format_input, parse_gt, unparse_gt

def infer_local_fetal_genotype(mother_gt, father_gt, cfdna_gt, cfdna_ad,
                               n_read, ff, tol = 0.001,
                               return_log = False, verbose = False):
    """Infer fetal genotype

    Model based on read counts for a single SNP, parental genotype,

    Potential input for genotypes:
        - ['x', 'y'] with x, y in {0,1}.
        - 'x/y' or 'x|y' with x, y in {0,1}.

    Potential input for allelic depth: np.array or list.

    Maternal and paternal genotype: '0/0', '0/1', '1/1'
    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele

    Input:
        mother_gt: maternal genotype.
        father_gt: paternal genotype.
        cfdna_gt: plasma genotype.
        cfdna_ad: vector of allele depth in plasma (read count per allele).
        n_read (int): total number of reads for cfdna_gt (coverage).
        ff (float): fetal fraction between 0 and 1.
        tol (float): tolerance for number comparison to zero. Default is 1e-3.
        return_log (boolean): if True, return un-normalized log-posteriors
            (i.e. joint log-likelihood), else return posterior probabilities.
            Default is False.
        verbose (bool): verbosity. Default is False.

    Output:
        prediction (string): predicted fetal genotype for the considered SNP
            by Maximum A Posteriori (MAP).
        posteriors (np.array): vector of posterior probability for each
            fetal genotype 0/0, 0/1, 1/0, 1/1 (in this order).
    """
    fetal_gt = ('', None)

    mother_gt, father_gt, cfdna_gt, cfdna_ad = format_input(
        mother_gt, father_gt, cfdna_gt, cfdna_ad)

    if verbose:
        print("----------------------------------------------------------------")

        print("mat = {} - pat = {} - cdna = {}"
                .format(unparse_gt(mother_gt), unparse_gt(father_gt),
                        unparse_gt(cfdna_gt, sort_out = False)))
        print("cfdna allelic depth = {}".format(cfdna_ad))
        print("fetal fraction = {}".format(ff))

    if snp_check(mother_gt, father_gt, cfdna_gt, cfdna_ad, n_read):
        ## priors on fetal genotypes
        fetal_gt_priors = fetal_genotype_prior(mother_gt, father_gt)
        if verbose:
            print("fetal gt priors = {}".format(fetal_gt_priors))
        ## data log-likelihood
        cfdna_data_loglikelihood = read_data_loglikelihood(
                                        mother_gt, father_gt,
                                        cfdna_gt, cfdna_ad, ff)
        if verbose:
            print("cfdna data loglike = {}".format(cfdna_data_loglikelihood))
        ## posteriors
        fetal_gt = model_posterior(cfdna_data_loglikelihood, fetal_gt_priors,
                                   tol, return_log)
        if verbose:
            print("predicted fetal genotype = {}".format(fetal_gt[0]))
            print("posterior = {}".format(fetal_gt[1]))

    return fetal_gt


def infer_global_fetal_genotype(seq_data_tab, fetal_fraction_tab,
                                min_coverage = 50, tol = 0.001,
                                snp_neighborhood = 1e5, n_neighbor_snp = 10,
                                return_log = False, verbose = False, **kwargs):
    """Infer the fetal genotypes along the genome

    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele

    Input:
        seq_data_tab (Pandas.DataFrame): sequencing data table produced
            by 'prediag.vcf_reader.load_vcf_data' function.
        fetal_fraction_tab (pandas.DataFrame): loci fetal fraction table
            produced by 'prediag.fetal_fraction.estimate_global_fetal_fraction'
            function.
        min_coverage (integer): minimum threshold for the coverage to
            consider the locus.
        tol (float): tolerance for number comparison to zero. Default is 1e-3.
        snp_neighborhood (float): maximum distance in bp to consider SNPs in the
            same neighborhood. Default value is 50e3.
        n_neighbor_snp (int): minimal number of SNPs in a neighborhood to
            consider using neighborhood average.
        verbose (bool): verbosity. Default is False.

    Output: Pandas.DataFrame with for each SNP
        * chrom (string): chromosome
        * pos (integer): position on the sequence.
        * mother_gt (string): maternal genotype 'x/y' with x, y in {0, 1}.
        * father_gt (string): maternal genotype 'x/y' with x, y in {0, 1}.
        * coverage (int): coverage on the locus.
        * fetal_fraction (float): estimated fetal fraction.
        * fetal_pred (string): inferred fetal genotype 'x/y' with x, y in
            {0, 1}.
        * fetal genotype posteriors (float vector of length 4): 'post_0/0',
            'post_0/1', 'post_1/0', 'post_1/1' respectively for genotypes
            '0/0', '0/1', '1/0', '1/1'.
    """
    # filter SNP table
    seq_data_tab = loci_tab_filter(seq_data_tab, min_coverage, verbose = True)

    # chromosome list (hash table)
    chrom_list = fetal_fraction_tab['chrom'].value_counts(sort=False)

    # iterate through sequencing data table and estimate fetal fraction
    out = []
    for index, row in seq_data_tab.iterrows():
        chrom = row["chrom"]
        pos = row["pos"]
        mother_gt = row["mother_gt"]
        father_gt = row["father_gt"]
        cfdna_gt = row["cfdna_gt"]
        cfdna_ad = row["cfdna_ad"]
        n_read = row["cfdna_dp"]

        # fetal fraction
        ff = average_fetal_fraction(chrom, pos, fetal_fraction_tab,
                                    snp_neighborhood, n_neighbor_snp,
                                    chrom_list)
        
        # prediction
        fetal_gt_pred, \
        fetal_gt_posteriors = infer_local_fetal_genotype(
                                    mother_gt, father_gt, cfdna_gt, cfdna_ad,
                                    n_read, ff, tol, return_log, verbose)

        out.append([chrom, pos, mother_gt, father_gt, cfdna_gt, cfdna_ad,
                    n_read, ff, fetal_gt_pred,
                    list(fetal_gt_posteriors) if fetal_gt_posteriors is not None
                    else fetal_gt_posteriors])

    post = 'fetal_gt_posterior'
    if return_log:
        post = 'fetal_gt_logposterior'

    df = pds.DataFrame(out, columns=['chrom', 'pos', 'mother_gt',
                                    'father_gt', 'cfdna_gt', 'cfdna_ad',
                                    'cfdna_dp', 'fetal_fraction',
                                    'fetal_gt_pred', post])
    return df


# example
if __name__ == '__main__':
    import itertools
    from prediag.fetal_fraction import estimate_local_fetal_fraction, estimate_global_fetal_fraction
    import prediag.simulation as simulation
    from prediag.utils import float2string

    # single SNP
    possible_gt = ['0/0', '0/1', '1/1']
    allele_origin = None
    ff = 0.2
    coverage = 100
    add_noise = False
    verbose = True
    tol = 0.05

    for mother_gt, father_gt in itertools.product(possible_gt, possible_gt):
        print("--------------------------------------------------------------")
        fetal_gt, cfdna_gt, cfdna_ad = simulation.single_snp_data(
                                        mother_gt, father_gt, allele_origin,
                                        ff, coverage, add_noise,
                                        verbose = False)
        # fetal fraction estimation
        n_read = np.sum(cfdna_ad)
        estim_ff = estimate_local_fetal_fraction(
                            mother_gt, father_gt,
                            cfdna_gt, cfdna_ad, n_read, tol)
        if estim_ff is None:
            estim_ff = ff

        # genotype inference
        prediction, posteriors = infer_local_fetal_genotype(
                            mother_gt, father_gt, cfdna_gt, cfdna_ad,
                            n_read, estim_ff, tol, return_log = False,
                            verbose = True)

    print("--------------------------------------------------------------")

    # multi SNP
    seq_length = 0.1
    snp_dist = 1e-3
    phased = True
    ff = 0.2
    ff_constant = False
    recombination_rate = 1.2e-8
    coverage = 100
    coverage_constant = False
    add_noise = True
    verbose = False

    simu_data = simulation.multi_snp_data(seq_length, snp_dist, phased,
                                          ff, ff_constant, recombination_rate,
                                          coverage, coverage_constant,
                                          add_noise, verbose)

    print("fetal fraction estimation")
    fetal_fraction_tab = estimate_global_fetal_fraction(simu_data, tol)
    print(fetal_fraction_tab.to_string(float_format = float2string))

    print("fetal genotype inference")
    fetal_genotype_tab = infer_global_fetal_genotype(
        simu_data, fetal_fraction_tab, min_coverage = 50, tol = 0.001,
        snp_neighborhood = 5e4, n_neighbor_snp = 10, return_log = False,
        verbose = False
    )
    print(fetal_genotype_tab.to_string(
        float_format = float2string,
        formatters = {'fetal_gt_posterior': float2string}
    ))

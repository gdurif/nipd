#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
from collections import Iterable
import numpy as np
import pandas as pds
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist
# internal
from prediag.filter import loci_tab_filter, snp_check
from prediag.utils import format_input, find_ad, is_het, parse_gt


def estimate_local_fetal_fraction(mother_gt, father_gt, cfdna_gt, cfdna_ad,
                                  n_read, tol=0.05):
    """Estimate fetal fraction from read counts for a single SNP

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
        tol (float): precision for read count comparison. Default is 5e-2.

    Input arguments `mother_gt`, `father_gt` and `cfdna_gt` can also be
    string "x/y" with x,y in {0,1}.

    Check in each scenario are detailed in project README.md file.

    Depending on maternal, paternal and cfDNA genotypes, allelic depth can give
    an insight on fetal genotype (c.f. project README.md file).

    Output (float): fetal fraction between 0 and 1.
    """
    ff = None

    mother_gt, father_gt, cfdna_gt, cfdna_ad = format_input(
        mother_gt, father_gt, cfdna_gt, cfdna_ad
    )

    if snp_check(mother_gt, father_gt, cfdna_gt, cfdna_ad, n_read):
        # case 1: mother A/A, father C/C
        # -> child A/C
        # -> plasma A/C
        if not is_het(mother_gt) and not is_het(father_gt) \
                and len(set(np.concatenate([mother_gt, father_gt]))) > 1 \
                and is_het(cfdna_gt):
            # read count
            N_A = find_ad(mother_gt[0], cfdna_gt, cfdna_ad)
            N_C = find_ad(father_gt[0], cfdna_gt, cfdna_ad)
            # check
            if (2 * N_C < n_read) and (N_A > N_C):
                ff = 2 * N_C / n_read
        # case 2: mother A/A, father A/C
        # -> child A/C (or A/A non informative)
        # -> plasma A/C (or A/A non informative)
        elif not is_het(mother_gt) and is_het(father_gt) \
                and is_het(cfdna_gt):
            # read count
            N_A = find_ad(mother_gt[0], cfdna_gt, cfdna_ad)
            N_C = find_ad(father_gt[father_gt != mother_gt[0]][0], cfdna_gt, cfdna_ad)
            # check
            if (2 * N_C < n_read) and (N_A > N_C):
                ff = 2 * N_C / n_read
        # case 3: mother A/C, father A/A
        # -> child A/A or A/C
        # -> plasma (A/C)
        elif is_het(mother_gt) and not is_het(father_gt) \
                and is_het(cfdna_gt):
            # read count
            N_A = find_ad(father_gt[0], cfdna_gt, cfdna_ad)
            N_C = find_ad(mother_gt[mother_gt != father_gt[0]][0], cfdna_gt, cfdna_ad)
            # check
            if (2 * N_C < n_read) and (N_A > N_C) and (abs(N_A - N_C) / n_read > tol):
                ff = 1 - 2 * N_C / n_read
        # case 4: mother A/C, father A/C
        # -> child A/A or C/C or A/C
        # -> plasma (A/C)
        elif is_het(mother_gt) and is_het(father_gt):
            # read count
            N_A = find_ad(mother_gt[0], cfdna_gt, cfdna_ad)
            N_C = find_ad(father_gt[father_gt != mother_gt[0]][0], cfdna_gt, cfdna_ad)
            # check
            if (abs(N_A - N_C) / n_read > tol):
                if (2 * N_C < n_read) and (N_A > N_C):
                    ff = 1 - 2 * N_C / n_read
                elif (2 * N_A < n_read) and (N_C > N_A):
                    ff = 1 - 2 * N_A / n_read

    return ff


def estimate_global_fetal_fraction(seq_data_tab, min_coverage = 50, tol = 0.05,
                                   **kwargs):
    """Estimate fetal fraction along the genome from VCF files

    Input:
        seq_data_tab (Pandas.DataFrame): sequencing data table produced
            by 'prediag.vcf_reader.load_vcf_data' function.
        min_coverage (integer): minimum threshold for the coverage to
            consider the locus.
        tol (float): precision for read count comparison. Default is 5e-2.

    Output: Pandas.DataFrame with following columns
        * chrom (string): chromosome
        * pos (integer): position on the sequence.
        * mother_gt (string): maternal genotype 'x/y' with x, y in {0, 1}.
        * father_gt (string): maternal genotype 'x/y' with x, y in {0, 1}.
        * cfdna_gt (string): cfDNA genotype 'x/y' with x, y in {0, 1}.
        * cfdna_ad (int list): allelic depths.
        * cfdna_dp (int): coverage.
        * cfdna_ff (float): estimated fetal fraction.
    """
    # filter SNP table
    seq_data_tab = loci_tab_filter(seq_data_tab, min_coverage, verbose = True)

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

        cfdna_ff = estimate_local_fetal_fraction(
            mother_gt, father_gt, cfdna_gt, cfdna_ad, n_read, tol
        )

        out.append([chrom, pos, mother_gt, father_gt, cfdna_gt, cfdna_ad,
                    n_read, cfdna_ff])

    df = pds.DataFrame(out, columns=['chrom', 'pos', 'mother_gt', 'father_gt',
                                     'cfdna_gt', 'cfdna_ad', 'cfdna_dp',
                                     'cfdna_ff'])
    return df


def impute_fetal_fraction(fetal_fraction_tab, max_na_prop = 0.9):
    """Impute missing fetal fraction

    Imputation with order-2 splines. If too many missing values, imputation
    with order-1 splines.

    Input:
        fetal_fraction_tab (Pandas.DataFrame): output of function
            `prediag.fetal_fraction.estimate_global_fetal_fraction`.
        max_na_prop (float): proportion between 0 and 1 of missing values per
            chromosome above which order-1 splines imputation is used.

    Output: `fetal_fraction_tab` with imputed missing values.
    """
    # group by chromosome
    tab_by_chrom = fetal_fraction_tab.groupby('chrom')
    # check proportion of missing values
    prop_na = tab_by_chrom['cfdna_ff'].apply(
        lambda group: group.isnull().sum()/len(group)
    )
    print("Proportion of missing values regarding ff by chromosome")
    print(prop_na)
    # if high missing value proportion
    if np.any(prop_na > max_na_prop):
        return tab_by_chrom.apply(
            lambda group: group.interpolate(
                method='spline', order=1, limit_direction='both'
            ) if group['cfdna_ff'].isnull().sum()/len(group) < 1 else group
        )
    # lower missing value proportion
    else:
        return tab_by_chrom.apply(
            lambda group: group.interpolate(
                method='spline', order=2, limit_direction='both'
            )
        )


def average_fetal_fraction(chrom, pos, fetal_fraction_tab,
                           snp_neighborhood = 1e5,
                           n_neighbor_snp = 10,
                           chrom_list = None):
    """Estimate fetal fraction in a neighborhood around a locus

    SNP neighborhood = 2 x `snp_neighborhood` around the locus `pos`.

    Fetal fraction estimated as:
        * average around the SNP position on the chromosome if possible
        (more than `n_neighbor_snp` loci in SNP neighborhood)
        * else average on the chromosome if possible.
        * else genome-wide average (no SNP on the chromosome)

    Input:
        chrom (string): targeted chromosome.
        pos (int): targeted locus on chromosome.
        fetal_fraction_tab (Pandas.DataFrame): output of function
            `prediag.fetal_fraction.infer_global_fetal_genotype_vcf`.
        snp_neighborhood (float): maximum distance in bp to consider SNPs in the
            same neighborhood. Default value is 1e5.
        n_neighbor_snp (int): minimal number of SNPs in a neighborhood to
            consider using neighborhood average.
        chrom_list (list of string): list of chromosome in the dataset.

    Return: estimated fetal fraction (float).
    """
    # drop na
    fetal_fraction_tab = fetal_fraction_tab.dropna()

    # check if any data remains
    if len(fetal_fraction_tab.index) == 0:
        raise ValueError("Fetal fraction cannot be estimated for any locus.")

    # chromosome list (hash table)
    if chrom_list is None:
        chrom_list = fetal_fraction_tab['chrom'].value_counts(sort=False)
    ## fetal fraction
    # - average around the SNP position on the chromosome if possible
    # - else average on the chromosome if possible
    # - else average
    valid_snp_mask = None
    if chrom in chrom_list.keys():
        # chromosome SNPs
        candidate_snp_mask = np.in1d(fetal_fraction_tab.chrom,
                                     np.array([chrom]))
        candidate_snp = fetal_fraction_tab[candidate_snp_mask]['pos']
        # SNP distances
        snp_dist = cdist(np.array(pos).reshape(1,1),
                         np.array(candidate_snp).reshape(-1,1)).reshape(-1)
        # neighbor SNPs
        valid_snp = snp_dist < snp_neighborhood
        # compute average fetal fraction in neighborhood if possible
        if np.sum(valid_snp) > n_neighbor_snp:
            valid_snp_mask = np.in1d(fetal_fraction_tab.pos,
                                     candidate_snp[valid_snp])
        else:
            valid_snp_mask = (fetal_fraction_tab.chrom == chrom)

    else:
        valid_snp_mask = np.repeat(True, len(fetal_fraction_tab.index))

    # average weighted by coverage
    weights = fetal_fraction_tab[valid_snp_mask]['cfdna_dp']
    weights = np.sum(valid_snp_mask) * weights / np.sum(weights)

    # average fetal fraction
    ff = np.average(fetal_fraction_tab[valid_snp_mask]['cfdna_ff'],
                    weights = weights)

    # output
    return ff


def smooth_fetal_fraction(fetal_fraction_tab):
    """Smoothing fetal fraction with order-2 spline model weighted by
    locus coverage.

    Input:
        fetal_fraction_tab (Pandas.DataFrame): output of function
            `prediag.fetal_fraction.estimate_global_fetal_fraction`.

    Output: `fetal_fraction_tab` with smoothed fetal fraction.
    """
    # non smoothed ff
    fetal_fraction_tab['cfdna_ff_not_smoothed'] = fetal_fraction_tab['cfdna_ff']

    # group by chromosome
    tab_by_chrom = fetal_fraction_tab.groupby('chrom')

    # smoothing function
    def apply_spline(tab):
        if tab['cfdna_ff'].isnull().sum()/len(tab) < 1:
            weights = tab['cfdna_dp']
            weights = len(tab.index) * weights / np.sum(weights)
            spline = UnivariateSpline(
                tab['pos'], tab['cfdna_ff'], w=weights,
                k=2, check_finite=True
            )
            tab['cfdna_ff'] = spline(tab['pos'])
        return tab

    # smoothing by chromosome
    fetal_fraction_tab = tab_by_chrom.apply(apply_spline)
    return fetal_fraction_tab


# example
if __name__ == '__main__':
    import itertools
    import prediag.simulation as simulation
    from prediag.utils import float2string

    # single SNP
    possible_gt = ['0/0', '0/1', '1/1']
    allele_origin = None
    ff = 0.2
    coverage = 100
    add_noise = False
    verbose = True
    min_coverage = 50
    tol = 0.05

    for mother_gt, father_gt in itertools.product(possible_gt, possible_gt):
        print("--------------------------------------------------------------")
        fetal_gt, cfdna_gt, cfdna_ad = simulation.single_snp_data(
            mother_gt, father_gt, allele_origin, ff, coverage, add_noise,
            verbose
        )

        # fetal fraction estimation
        n_read = np.sum(cfdna_ad)
        estim_ff = estimate_local_fetal_fraction(
            mother_gt, father_gt, cfdna_gt, cfdna_ad, n_read, tol
        )

        if estim_ff is None:
            print("estimated ff = {} -- true ff = {}"
                  .format(float2string(estim_ff), float2string(ff)))

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

    simu_data = simulation.multi_snp_data(
        seq_length, snp_dist, phased, ff, ff_constant, recombination_rate,
        coverage, coverage_constant, add_noise, verbose
    )

    print("fetal fraction estimation")
    fetal_fraction_tab = estimate_global_fetal_fraction(simu_data, min_coverage, tol)
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

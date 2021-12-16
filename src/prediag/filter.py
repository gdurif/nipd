#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
from collections import Iterable
import numpy as np
# internal
from prediag.utils import is_phased, is_polyallelic, parse_gt


def snp_check(mother_gt, father_gt, cfdna_gt, cfdna_ad, cfdna_dp = None,
              min_coverage = None):
    """Filter out useless SNP

    A genotype is a {0,1} valued vector of allele

    Input:
        mother_gt (np.array): vector of maternal alleles.
        father_gt (np.array): vector of paternal alleles.
        cfdna_gt (np.array): vector of plasma alleles.
        cfdna_ad (np.array): vector of allelic depth in plasma.
        cfdna_dp (int): cfDNA coverage (= total read count) on the locus.
        min_coverage (integer): minimum threshold for the coverage to
            consider the locus. None to bypass this check.

    Output: boolean value, true to keep the SNP, false to discard it.
    """
    out = True
    # fitler out SNP without allelic depth
    if cfdna_ad is None:
        out = False
    # filter out SNP with low coverage
    elif min_coverage is not None and cfdna_dp is not None and cfdna_dp < min_coverage:
        out = False
    # filter "impossible" SNP where parents = x/x and cfdna = x/y
    elif not np.all(np.isin(cfdna_gt, np.concatenate([mother_gt, father_gt]))):
        out = False
    # filter "impossible" SNP where mother = x/y and cfdna = x/x
    elif not np.all(np.isin(mother_gt, cfdna_gt)):
        out = False
    # filter out poly-allelic locus
    elif is_polyallelic(mother_gt) or is_polyallelic(father_gt) or \
        is_polyallelic(cfdna_gt):
        out = False
    # output
    return out


def snp_filter(mother_gt, father_gt, cfdna_gt, cfdna_ad, cfdna_dp,
               min_coverage = None):
    """Filter out useless SNP

    A genotype is a {0,1} valued vector of allele

    Input:
        mother_gt (string): maternal haplotype 'x|y' with x, y in {0,1},
            or maternal genotype 'x/y' if haplotype not available.
        father_gt (string): paternal haplotype 'x|y' with x, y in {0,1},
            or paternal genotype 'x/y' if haplotype not available.
        cfdna_gt (string): plasma (=cfDNA) genotype, i.e. 'x/y' with x, y
            in {0, 1}.
        cfdna_ad (int list): cfDNA allelic depth (= read count) per allele.
        cfdna_dp (int): cfDNA coverage (= total read count) on the locus.
        min_coverage (integer): minimum threshold for the coverage to
            consider the locus. None to bypass this check.

    Output: boolean value, true to keep the SNP, false to discard it.
    """
    out = True
    # filter out SNP with missing information
    check = [item is None or
                (isinstance(item, Iterable) and len(item) == 0)
                for item
                in [mother_gt, father_gt, cfdna_gt, cfdna_ad, cfdna_dp]]
    if np.any(check):
        out = False
    # filter out SNP with low coverage
    elif min_coverage is not None and cfdna_dp < min_coverage:
        out = False
    # filter "impossible" SNP, i.e. parents = x/x and cfdna = x/y
    elif not np.all(np.isin(
                parse_gt(cfdna_gt),
                np.concatenate([parse_gt(mother_gt), parse_gt(father_gt)]))):
        out = False
    # filter "impossible" SNP where mother = x/y and cfdna = x/x
    elif not np.all(np.isin(parse_gt(mother_gt), parse_gt(cfdna_gt))):
        out = False
    # filter out poly-allelic locus
    elif is_polyallelic(mother_gt) or is_polyallelic(father_gt) or \
        is_polyallelic(cfdna_gt):
        out = False
    # output
    return out


def hap_filter(mother_hap, father_hap, mother_pq, mother_jq, father_pq,
               father_jq):
    """Filter phased haplotypes

    Input:
        mother_hap (string): maternal haplotype 'x|y' with x, y in {0,1},
            or maternal genotype 'x/y' if haplotype not available.
        father_hap (string): paternal haplotype 'x|y' with x, y in {0,1},
            or paternal genotype 'x/y' if haplotype not available.
        mother_pq (float): mother phasing quality probability, "probability
            that alleles are phased incorrectly in a heterozygous call"
            (10x-genomics doc).
        mother_jq (float): mother junction quality probability, "probability
            that there is a large-scale phasing switch error occuring between
            this variant and the following variant" (10x-genomics doc).
        father_pq (float): father phasing quality probability, "probability
            that alleles are phased incorrectly in a heterozygous call"
            (10x-genomics doc).
        father_jq (float): father junction quality probability, "probability
            that there is a large-scale phasing switch error occuring between
            this variant and the following variant" (10x-genomics doc).
    """
    out = True
    # mother
    if is_phased(mother_hap) and (mother_pq is None or mother_jq is None or
                                  np.any(np.isnan([mother_pq, mother_jq]))):
        out = False
    # father
    if is_phased(father_hap) and (father_pq is None or father_jq is None or
                                  np.any(np.isnan([father_pq, father_jq]))):
        out = False
    # output
    return out


def loci_tab_filter(loci_tab, min_coverage = 50, verbose = False):
    """Filter table of SNPs

    Input:
        loci_tab (Pandas.DataFrame): table of loci with at least the following
            fields:
            * chrom (string): chromosome
            * pos (integer): position on the sequence.
            * mother_gt (string): maternal haplotype 'x|y' with x, y in {0,1},
                or maternal genotype 'x/y' if haplotype not available.
            * father_gt (string): paternal haplotype 'x|y' with x, y in {0,1},
                or paternal genotype 'x/y' if haplotype not available.
            * cfdna_gt (string): plasma (=cfDNA) genotype, i.e. 'x/y' with x, y
                in {0, 1}.
            * cfdna_ad (int list): cfDNA allelic depth (= read count) per allele.
            * cfdna_dp (int): cfDNA coverage (= total read count) on the locus.
        min_coverage (integer): minimum threshold for the coverage to
            consider the locus.
    """
    # filter SNPs with issues
    mask1 = loci_tab.apply(
        lambda row: snp_filter(
            row.mother_gt, row.father_gt, row.cfdna_gt, row.cfdna_ad,
            row.cfdna_dp, min_coverage
        ), axis=1
    )
    # filter haplotypes with issues
    mask2 = loci_tab.apply(
        lambda row: hap_filter(
            row.mother_gt, row.father_gt,
            row.mother_pq, row.mother_jq,
            row.father_pq, row.father_jq
        ), axis=1
    )
    # verbosity
    if verbose:
        print("Filtering: {} valid loci over {} total loci"
                .format(np.sum(np.logical_and(mask1, mask2)),
                        len(loci_tab.index)))
    # output
    return loci_tab[np.logical_and(mask1, mask2)]

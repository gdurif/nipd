#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
from collections import Iterable
import numpy as np
import re
import scipy.special
from warnings import warn as warning


def float2string(input):
    """convert float to string for printing

    Input (float)
    Output (string)
    """
    if isinstance(input, Iterable):
        return list(map(float2string, input))
    else:
        if input is None:
            return None
        else:
            if float(input).is_integer():
                return "{}".format(input)
            if abs(input) < 1e-2 or abs(input) > 1e2:
                return "{:.2e}".format(input)
            else:
                return "{:.3f}".format(input)


def format_input(mother_gt, father_gt, cfdna_gt, cfdna_ad):
    """Format input data (genotypes and allelic depth)

    Potential input for genotypes:
        - ['x', 'y'] with x, y in {0,1}.
        - 'x/y' or 'x|y' with x, y in {0,1}.

    Potential input for allelic depth: np.array or list.

    Input:
        mother_gt: maternal genotype.
        father_gt: paternal genotype.
        cfdna_gt: plasma genotype.
        cfdna_ad: vector of allele depth in plasma (read count per allele).

    Output:
        mother_gt: maternal genotype ['x', 'y'] with x,y in {0,1}.
        father_gt: paternal genotype ['x', 'y'] with x,y in {0,1}.
        cfdna_gt: plasma genotype ['x', 'y'] with x,y in {0,1}.
        cfdna_ad (np.array): vector of allele depth in plasma (read count
            per allele).
    """
    if isinstance(mother_gt, str):
        mother_gt = parse_gt(mother_gt)
    elif not isinstance(mother_gt, np.ndarray):
        mother_gt = np.array(mother_gt)
    if isinstance(father_gt, str):
        father_gt = parse_gt(father_gt)
    elif not isinstance(father_gt, np.ndarray):
        father_gt = np.array(father_gt)
    if isinstance(cfdna_gt, str):
        cfdna_gt = parse_gt(cfdna_gt)
    elif not isinstance(cfdna_gt, np.ndarray):
        cfdna_gt = np.array(cfdna_gt)
    if cfdna_ad is not None and not isinstance(cfdna_ad, np.ndarray):
        cfdna_ad = np.array(cfdna_ad)

    return mother_gt, father_gt, cfdna_gt, cfdna_ad


def find_ad(target, cfdna_gt, cfdna_ad):
    """Find allelic depth of target allele in cfdna genotype

    Input:
        target (string): target allele in {'0', '1'}.
        cfdna_gt (string np.array): list of cfdna haplotype, i.e. ['x', 'y'] with
            x, y in {0, 1}.
        cfdna_ad (int np.array): list of allelic depth in `cfdna_gt`.

    Output: index of parent_hap in cfdna_gt

    >>> find_ad('0', ['0', '1'], [10, 40])
    10
    >>> find_ad('1', ['0', '1'], [10, 40])
    40
    >>> find_ad('0', ['1', '1'], [0, 40])
    0
    >>> find_ad('1', ['1', '1'], [0, 40])
    40
    >>> find_ad('0', ['0', '0'], [40, 0])
    40
    >>> find_ad('1', ['0', '0'], [40, 0])
    0
    """
    if not isinstance(cfdna_ad, np.ndarray):
        cfdna_ad = np.array(cfdna_ad)
    if not isinstance(cfdna_gt, np.ndarray):
        cfdna_gt = np.array(cfdna_gt)
    if target not in cfdna_gt:
        return 0
    else:
        return np.max(cfdna_ad[cfdna_gt == np.repeat(target, len(cfdna_gt))])


def is_het(gt):
    """Check if a genotype is heterozygous

    Genotypes are assumed to be represented as a character list ['x', 'y']
    where x, y in {0, 1}.

    Input:
        gt (character list or np.array): list of two homologous haplotypes,
            i.e ['x', 'y'] where x, y in {0, 1}.

    Output: boolean, `x == y`.

    >>> is_het(['0', '0'])
    False
    >>> is_het(['0', '1'])
    True
    """
    return gt[0] != gt[1]


def is_phased(gt):
    """Check if a genotype is phased

    Genotypes are assumed to be represented as a string 'x/y' (unphased) or
    'x|y' (phased) where x, y in {0, 1}.

    Input:
        gt (string): genotype 'x/y' (unphased) or 'x|y' (phased) where
            x, y in {0, 1}.

    Output: boolean.

    >>> is_phased('x|y')
    True
    >>> is_phased('x/y')
    False
    """
    if not isinstance(gt, str):
        raise ValueError("gt input parameter should be a genotype string 'x|y' or 'x/y'.")

    return bool(re.search(r"\|", gt))


def is_polyallelic(gt):
    """Check if the genotype correspond to a poly(>2)-allelic locus or
    not (i.e. mono or bi-allelic).

    Input:
        gt (string): genotype 'x/y' (unphased) or 'x|y' (phased) where
            x, y in {0, 1}.

    Output: boolean

    >>> is_polyallelic('1|2')
    True
    >>> is_polyallelic('0/1')
    False
    """
    if isinstance(gt, str):
        gt = parse_gt(gt)

    if not np.all([a in ['0', '1'] for a in gt]):
        warning(
            "Poly-allelic SNPs (i.e. with more than 2 alleles: "
            + "0,1,2,...) are discarded for the moment."
        )
        return True
    else:
        return False


def parse_allele_origin(allele_origin):
    """Parse allele origin to index pair

    Input: index allele origin 'x-y' with x, y in {0, 1}.
    Output: pair of integer index [x, y] (np.array).

    If x or y is missing, it is replaced by None.

    >>> list(parse_allele_origin('0-1'))
    [0, 1]
    """
    if not isinstance(allele_origin, str) or \
        not re.match(r"^[0-9]-[0-9]$", allele_origin):
        raise ValueError("Issue with 'allele_origin' value")

    allele_origin1 = re.findall("[0-9](?=-)", allele_origin)
    allele_origin2 = re.findall("(?<=-)[0-9]", allele_origin)

    return np.array([int(allele_origin1[0]) if len(allele_origin1) > 0 else None,
                     int(allele_origin2[0]) if len(allele_origin2) > 0 else None])


def unparse_allele_origin(allele_origin):
    """Unparse allele origin to index pair

    Input: pair of integer index [x, y] (int np.array).
    Output: index allele origin 'x-y' with x, y in {0, 1}.

    x or y can be missing if information is None in input.

    >>> unparse_allele_origin([0, 1])
    '0-1'
    """
    if not isinstance(allele_origin, np.ndarray):
        allele_origin = np.array(allele_origin)
    sep = '-'
    allele_origin = np.vectorize(lambda x:
                            str(x) if x is not None else '')(allele_origin)
    return sep.join(allele_origin)


def readable_allele_origin(allele_origin):
    """Transform allele origin index to human readable allele origin

    Input: index allele origin 'x-y' with x, y in {0, 1}.
    Output: corresponding allele origin 'matA-patB' with A, B in {1, 2}.

    Index 0 = haplotype 1
    Index 1 = haplotype 2

    If x or y is missing, it is replaced by None.

    >>> readable_allele_origin('0-1')
    'mat1-pat2'
    """
    if not isinstance(allele_origin, str) or \
        not re.match(r"^[0-9]?-[0-9]?$", allele_origin):
        raise ValueError("Issue with 'allele_origin' value")

    allele_origin = re.sub(r'1', '2', allele_origin)
    allele_origin = re.sub(r'0', '1', allele_origin)

    allele_origin1 = re.findall("[0-9](?=-)", allele_origin)
    allele_origin2 = re.findall("(?<=-)[0-9]", allele_origin)

    return (("mat" + allele_origin1[0]) if len(allele_origin1) > 0 else '') \
            + '-' \
            + (("pat" + allele_origin2[0]) if len(allele_origin2) > 0 else '')


def index_allele_origin(allele_origin):
    """Transform human readable allele origin to allele origin index

    Input: allele origin 'matA-patB' with A, B in {1, 2}.
    Output: corresponding index allele origin 'x-y' with x, y in {0, 1}.

    Index 0 = haplotype 1
    Index 1 = haplotype 2

    If x or y is missing, it is replaced by None.

    >>> index_allele_origin('mat1-pat2')
    '0-1'
    """
    if not isinstance(allele_origin, str) or \
        not re.match(r"^(mat[0-9])?-(pat[0-9])?$", allele_origin):
        raise ValueError("Issue with 'allele_origin' value")

    allele_origin = re.sub(r'1', '0', allele_origin)
    allele_origin = re.sub(r'2', '1', allele_origin)

    allele_origin1 = re.findall("[0-9](?=-)", allele_origin)
    allele_origin2 = re.findall("(?<=-pat)[0-9]", allele_origin)

    return (allele_origin1[0] if len(allele_origin1) > 0 else '') \
            + '-' \
            + (allele_origin2[0] if len(allele_origin2) > 0 else '')


def parse_gt(gt):
    """Parse genotype to haplotype pair

    Input: genotype "x/y" or "x|y" (string) with x, y in {0, 1}.
    Output: pair of corresponding haplotype ['x', 'y'] (character list).

    >>> list(parse_gt('0/1'))
    ['0', '1']
    >>> list(parse_gt('0|1'))
    ['0', '1']
    """
    return np.array(re.findall("[0-9]", gt))


def unparse_gt(gt, sort_out = True, phased = False):
    """Unparse haplotype pair to genotype

    Note: phased haplotypes are not sorted.

    Input:
        gt (character list or np.array): list of two homologous haplotypes,
            i.e ['x', 'y'] where x, y in {0, 1}.
        sort_out (bool): should the genotypes be sorted in output. Default
            is False.
        phased (bool): return haplotypes (separated by '|') if True, or
            genotypes (separated by '/') if False. Default value is False.
    Output (string): genotypes 'x/y' or haplotypes 'x|y' with x, y in {0, 1}.

    >>> unparse_gt(['0', '1'])
    '0/1'
    >>> unparse_gt(['1', '1'])
    '1/1'
    """
    sep = '/'
    if phased:
        sep = '|'
        sort_out = False
    if sort_out:
        return sep.join(np.sort(gt))
    else:
        return sep.join(gt)


def softmax(vec):
    """Compute softmax function from input vector

    Input:
        vec (np.array): vector of activation.

    Output: corresponding vector of probability according to softmax.

    >>> softmax([1,1,1])
    array([0.33333333, 0.33333333, 0.33333333])
    """
    return scipy.special.softmax(vec)


def round0(vec):
    """Round small values

    If 0 < input < 1e-12 then output = 1e-12 else output = input
    """
    return np.where(vec > 1e-12, vec, 1e-12)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
import numpy as np
import re
from scipy.special import softmax
import warnings
# internal
from prediag.filter import snp_check
import prediag.snp_model as snp_model
from prediag.utils import find_ad, float2string, format_input, is_het, is_phased, parse_gt, parse_allele_origin


def fetal_genotype_prior(mother_hap, father_hap, allele_origin):
    """Compute fetal genotype prior according to Mendelian law for a given
    locus, knowing parental haplotypes (or genotypes) and fetal allele origin
    in parental haplotypes for current locus.

    Maternal and paternal haplotype: '0|0', '0|1', '1|0' '1|1'

    If parental genotype are not phased (no haplotype available): parental
    haplotypes are replaced by parental genotypes: '0/0', '0/1', '1/1'.

    Fetal allele origin in parental haplotypes:
        - '0-0', '0-1', '1-0', '1-1' (lexicographic order)
        - with following convention: 'A-B' where A = maternal haplotype origin,
        and B = paternal haplotype origin, 0 = haplotype 1 and 1 = haplotype 2,
        i.e. if parental haplotype = 'x|y', 0 corresponds to x, and 1 to y.

    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele.

    Input:
        mother_hap (string): maternal haplotype 'x|y' with x, y in {0,1},
            or maternal genotype 'x/y' if haplotype not available.
        father_hap (string): paternal haplotype 'x|y' with x, y in {0,1},
            or paternal genotype 'x/y' if haplotype not available.
        locus (string): fetal allele origin in parental haplotypes 'a-b'
            with a, b in {0,1}, for current locus.
        allele_origin (string): fetal allele origin in parental haplotypes
            'a-b' with a, b in {0,1}.

    Output: vector of prior probability for each fetal genotype '0/0', '0/1',
    '1/0', '1/1' (in this order).
    """
    out = None

    if not isinstance(mother_hap, str) or not isinstance(father_hap, str):
        raise ValueError("gt input parameter should be a genotype string 'x|y' or 'x/y'.")

    if isinstance(allele_origin, str):
        allele_origin = parse_allele_origin(allele_origin)

    # parental genotype
    mother_gt = parse_gt(mother_hap)
    father_gt = parse_gt(father_hap)

    # parental allele allele origin
    mat_allele_origin = allele_origin[0]
    pat_allele_origin = allele_origin[1]

    # possible fetal genotypes
    possible_fetal_gt = np.array(['0/0', '0/1', '1/0', '1/1'])

    # maternal possible allele
    mat_allele = mother_gt
    if mat_allele_origin is not None and is_phased(mother_hap):
        mat_allele = list(mother_gt[mat_allele_origin])

    # paternal possible allele
    pat_allele = father_gt
    if pat_allele_origin is not None and is_phased(father_hap):
        pat_allele = list(father_gt[pat_allele_origin])

    # fetal gt
    fetal_gt = np.unique(['{}/{}'.format(mat, pat) for mat in mat_allele
                                                    for pat in pat_allele])

    # update fetal gt prior
    out = np.array([0., 0., 0., 0.])
    out[np.where(np.isin(possible_fetal_gt, fetal_gt))] = 1./len(fetal_gt)

    # output
    return out


def read_data_likelihood(mother_hap, father_hap, cfdna_gt, cfdna_ad, ff,
                         verbose = False):
    """Compute reads likelihood knowing fetal allele origin among parental
    haplotype for a SNP

    Maternal and paternal haplotype: '0|0', '0|1', '1|0' '1|1'

    If parental genotype are not phased (no haplotype available): parental
    haplotypes are replaced by parental genotypes: '0/0', '0/1', '1/1'.

    Fetal allele origin in parental haplotypes:
        - '0-0', '0-1', '1-0', '1-1' (lexicographic order)
        - with following convention: 'A-B' where A = maternal haplotype origin,
        and B = paternal haplotype origin, 0 = haplotype 1 and 1 = haplotype 2,
        i.e. if parental haplotype = 'x|y', 0 corresponds to x, and 1 to y.

    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele.

    Potential input for allelic depth: np.array or list.

    Input:
        mother_hap (string): maternal haplotype 'x|y' with x, y in {0,1},
            or maternal genotype 'x/y' if haplotype not available.
        father_hap (string): paternal haplotype 'x|y' with x, y in {0,1},
            or paternal genotype 'x/y' if haplotype not available.
        cfdna_gt (string): plasma genotype, i.e. 'x/y' with x, y in {0, 1}.
        cfdna_ad: vector of allele depth in plasma (read count per allele).
        ff (float): fetal fraction between 0 and 1.
        verbose (bool): set verbosity.

    Output: vector of data probability knowing each fetal allele origin
        '0-0', '0-1', '1-0', '1-1' (in this order).
    """
    out = None

    mother_gt, father_gt, cfdna_gt, cfdna_ad = format_input(
            mother_hap, father_hap, cfdna_gt, cfdna_ad
    )

    if snp_check(mother_gt, father_gt, cfdna_gt, cfdna_ad):
        ## number of reads per allele
        N_0 = find_ad('0', cfdna_gt, cfdna_ad)
        N_1 = find_ad('1', cfdna_gt, cfdna_ad)
        ## read conditional likelihood knowing fetal genotype
        # read 0
        read0_cond_loglikelihood = snp_model.read_data_loglikelihood(
                            mother_hap, father_hap, cfdna_gt,
                            cfdna_ad, ff, single = True, read_val = 0)
        if verbose:
            print("-- read0_cond_loglikelihood = {}"
                  .format(float2string(read0_cond_loglikelihood)))

        # read 1
        read1_cond_loglikelihood = snp_model.read_data_loglikelihood(
                            mother_hap, father_hap, cfdna_gt,
                            cfdna_ad, ff, single = True, read_val = 1)
        if verbose:
            print("-- read1_cond_loglikelihood = {}"
                  .format(float2string(read1_cond_loglikelihood)))

        # joint conditional loglikelihood
        read_cond_loglikelihood = np.array([0., 0., 0., 0.])
        if N_0 > 0:
            read_cond_loglikelihood += N_0 * read0_cond_loglikelihood
        if N_1 > 0:
            read_cond_loglikelihood += N_1 * read1_cond_loglikelihood
        # conditional likelihood
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            read_cond_likelihood = softmax(read_cond_loglikelihood)

        if verbose:
            print("-- read_cond_likelihood = {}"
                  .format(float2string(read_cond_likelihood)))

        ## knowing allele origin
        out = []
        for allele_origin in ['0-0', '0-1', '1-0', '1-1']:
            if verbose:
                print("---- allele origin = {}".format(allele_origin))

            # fetal genotype prior
            fetal_gt_prior = fetal_genotype_prior(mother_hap, father_hap,
                                                  allele_origin)
            if verbose:
                print("---- fetal_gt_prior = {}"
                      .format(float2string(fetal_gt_prior)))

            # data likelihood
            data_likelihood = np.sum(np.multiply(read_cond_likelihood,
                                                 fetal_gt_prior))
            if verbose:
                print("-- data likelihood = {}"
                      .format(float2string(data_likelihood)))

            # joint probabilities
            out.append(data_likelihood)
        # convert to np array
        out = np.array(out)
        if verbose:
            print("-- data likelihood = {}".format(float2string(out)))
    # output
    return out


def haplotype_transition_probability(previous_allele_origin,
                                     mother_phasing_error_proba,
                                     father_phasing_error_proba,
                                     locus_dist, recombination_rate = 1.2e-8,
                                     mother_phased = True,
                                     father_phased = True):
    """Compute probability of fetal allele origin in parental haplotypes for
    a locus given previous locus fetal allele origin.

    Fetal allele origin in parental haplotypes:
        - '0-0', '0-1', '1-0', '1-1' (lexicographic order)
        - with following convention: 'A-B' where A = maternal haplotype origin,
        and B = paternal haplotype origin, 0 = haplotype 1 and 1 = haplotype 2,
        i.e. if parental haplotype = 'x|y', 0 corresponds to x, and 1 to y.

    If a parental locus is not phased, corresponding switch probability
    becomes 1/2.

    Input:
        previous_allele_origin (string): previous locus fetal allele origin
            in parental haplotypes, i.e. 'A-B' with A, B in {0, 1}. If None,
            first locus of the sequence and no transition with previous locus,
            only phasing error is accounted for.
        mother_phasing_error_proba (float): probability of phasing at current
            locus in maternal haplotype.
        father_phasing_error_proba (float): probability of phasing at current
            locus in paternal haplotype.
        locus_dist (float): distance in bp between current and previous loci.
        recombination_rate (float): recombination rate per bp (between 0 and 1).
            Default value 1.2e-8.
        mother_phased (bool): indicator if current locus is phased (True) or
            not (False) in mother haplotypes. Default value is True.
        father_phased (bool): indicator if current locus is phased (True) or
            not (False) in father haplotypes. Default value is True.

    Output: vector of transition probability for each fetal allele origin
        '0-0', '0-1', '1-0', '1-1' (in this order).
    """

    # if no allele origin information for previous locus
    if previous_allele_origin is None:
        return np.array([0.25, 0.25, 0.25, 0.25])

    # else
    if previous_allele_origin is not None \
        and isinstance(previous_allele_origin, str):
        previous_allele_origin = parse_allele_origin(previous_allele_origin)

    # recombination probability
    recomb_proba = recombination_rate * locus_dist

    ## switch probabilities
    mother_switch_proba = \
        recomb_proba * (1 - mother_phasing_error_proba) \
        + (1 - recomb_proba) * mother_phasing_error_proba \
        if mother_phased and previous_allele_origin[0] is not None else 0.5
    father_switch_proba = \
        recomb_proba * (1 - father_phasing_error_proba) \
        + (1 - recomb_proba) * father_phasing_error_proba \
        if father_phased and previous_allele_origin[1] is not None else 0.5

    ## transition probabilities
    # '0-0'
    pr00 = (int(previous_allele_origin[0] == 0) * (1 - mother_switch_proba)
                + int(previous_allele_origin[0] == 1 or previous_allele_origin[0] is None)
                    * mother_switch_proba) \
            * (int(previous_allele_origin[1] == 0) * (1 - father_switch_proba)
                + int(previous_allele_origin[1] == 1 or previous_allele_origin[1] is None)
                    * father_switch_proba)
    # '0-1'
    pr01 = (int(previous_allele_origin[0] == 0) * (1 - mother_switch_proba)
                + int(previous_allele_origin[0] == 1 or previous_allele_origin[0] is None)
                    * mother_switch_proba) \
            * (int(previous_allele_origin[1] == 1) * (1 - father_switch_proba)
                + int(previous_allele_origin[1] == 0 or previous_allele_origin[1] is None)
                    * father_switch_proba)
    # '1-0'
    pr10 = (int(previous_allele_origin[0] == 1) * (1 - mother_switch_proba)
                + int(previous_allele_origin[0] == 0 or previous_allele_origin[0] is None)
                    * mother_switch_proba) \
            * (int(previous_allele_origin[1] == 0) * (1 - father_switch_proba)
                + int(previous_allele_origin[1] == 1 or previous_allele_origin[1] is None)
                    * father_switch_proba)
    # '1-1'
    pr11 = (int(previous_allele_origin[0] == 1) * (1 - mother_switch_proba)
                + int(previous_allele_origin[0] == 0 or previous_allele_origin[0] is None)
                    * mother_switch_proba) \
            * (int(previous_allele_origin[1] == 1) * (1 - father_switch_proba)
                + int(previous_allele_origin[1] == 0 or previous_allele_origin[1] is None)
                    * father_switch_proba)

    return np.array([pr00, pr01, pr10, pr11])


def model_conditional_posterior(mother_hap, father_hap, cfdna_gt, cfdna_ad, ff,
                                previous_allele_origin,
                                mother_phasing_error_proba,
                                father_phasing_error_proba,
                                locus_dist, recombination_rate = 1.2e-8,
                                mother_phased = True, father_phased = True,
                                verbose = False):
    """Compute fetal allele origin in parental haplotypes conditional posterior,
    i.e. the posterior for a locus i, given the locus i-1 and the data

    l_i = parental allele origin 'A-B' at fetal locus i

    $$p(l_i | data, l_{i-1}) ~ p(data at locus i | l_i) * p(l_i | l_{i-1})$$

    - $p(data at locus i | l_i) = \prod_j p(read_j at locus i | l_i)$
    - $p(l_i | l_{i-1})$ = transition probabilities (depends on reconbination
      probability between locus i and i-1, and phasing error probability at
      locus i)

    - Read-wise likelihood:
    $$p(read_j at locus i | l_i)
        = \sum_{fetal gt} p(read_j | fetal gt, mother gt) *
                             p(fetal gt | mother gt, father gt, l_i)$$

    Maternal and paternal haplotype: '0|0', '0|1', '1|0' '1|1'

    If parental genotype are not phased (no haplotype available): parental
    haplotypes are replaced by parental genotypes: '0/0', '0/1', '1/1'.

    Fetal allele origin in parental haplotypes:
        - '0-0', '0-1', '1-0', '1-1' (lexicographic order)
        - with following convention: 'A-B' where A = maternal haplotype origin,
        and B = paternal haplotype origin, 0 = haplotype 1 and 1 = haplotype 2,
        i.e. if parental haplotype = 'x|y', 0 corresponds to x, and 1 to y.

    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele.

    Input:
        mother_hap (string): maternal haplotype 'x|y' with x, y in {0,1},
            or maternal genotype 'x/y' if haplotype not available.
        father_hap (string): paternal haplotype 'x|y' with x, y in {0,1},
            or paternal genotype 'x/y' if haplotype not available.
        cfdna_gt (string): plasma genotype, i.e. 'x/y' with x, y in {0, 1}.
        cfdna_ad: vector of allele depth in plasma (read count per allele).
        ff (float): fetal fraction between 0 and 1.
        previous_allele_origin (string): previous locus fetal allele origin
            in parental haplotypes, i.e. 'A-B' with A, B in {0, 1}. If None,
            first locus of the sequence and no transition with previous locus,
            only phasing error is accounted for.
        mother_phasing_error_proba (float): probability of phasing at current
            locus in maternal haplotype.
        father_phasing_error_proba (float): probability of phasing at current
            locus in paternal haplotype.
        locus_dist (float): distance in bp between current and previous loci.
        recombination_rate (float): recombination rate per bp (between 0 and 1).
            Default value 1.2e-8.
        mother_phased (bool): indicator if current locus is phased (True) or
            not (False) in mother haplotypes. Default value is True.
        father_phased (bool): indicator if current locus is phased (True) or
            not (False) in father haplotypes. Default value is True.
        verbose (bool): set verbosity.

    Output:
        posteriors (np.array): vector of posterior probability for each
            fetal allele origin '0-0', '0-1', '1-0', '1-1' (in this order).
    """
    out = None
    # read data likelihood
    data_likelihood = read_data_likelihood(mother_hap, father_hap, cfdna_gt,
                                           cfdna_ad, ff, verbose = False)
    if verbose:
        print("data likelihood = {}".format(float2string(data_likelihood)))
    # transition probability
    if verbose:
        print("previous_allele_origin = {}".format(previous_allele_origin))
        print("mother_phasing_error_proba = {}".format(mother_phasing_error_proba))
        print("father_phasing_error_proba = {}".format(father_phasing_error_proba))
        print("locus_dist = {}".format(locus_dist))
        print("recombination_rate = {}".format(recombination_rate))
        print("mother_phased = {}".format(mother_phased))
        print("father_phased = {}".format(father_phased))

    trans_prob = haplotype_transition_probability(
                            previous_allele_origin,
                            mother_phasing_error_proba,
                            father_phasing_error_proba,
                            locus_dist, recombination_rate,
                            mother_phased, father_phased)
    if verbose:
        print("transition probabilities = {}".format(float2string(trans_prob)))
    # log-probability
    if data_likelihood is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = softmax(np.log(data_likelihood) + np.log(trans_prob))

    if verbose:
        print("proba = {}".format(out))

    return out


# example
if __name__ == '__main__':
    import itertools
    from prediag.fetal_fraction import estimate_local_fetal_fraction
    import prediag.simulation as simulation

    ## priors
    possible_haplotype = ['0|0', '0|1', '1|0', '1|1']
    possible_allele_origin = ['0-0', '0-1', '1-0', '1-1']
    possible_fetal_gt = np.array(['0/0', '0/1', '1/0', '1/1'])
    # phased haplotype
    for mother_hap, father_hap, allele_origin  in itertools.product(possible_haplotype,
                                                                    possible_haplotype,
                                                                    possible_allele_origin):
        allele_origin = parse_allele_origin(allele_origin)
        expected_val = np.array([0., 0., 0., 0.])
        fetal_gt = parse_gt(mother_hap)[allele_origin[0]] + '/' + parse_gt(father_hap)[allele_origin[1]]
        expected_val[np.where(possible_fetal_gt == fetal_gt)] = 1.
        assert np.all(fetal_genotype_prior(mother_hap, father_hap, allele_origin) == expected_val)

    # phased mum
    assert np.all(fetal_genotype_prior('0|0', '0/0', '0-0') == np.array([1., 0, 0, 0]))
    assert np.all(fetal_genotype_prior('0|0', '0/1', '0-0') == np.array([0.5, 0.5, 0, 0]))
    assert np.all(fetal_genotype_prior('0|0', '1/1', '0-0') == np.array([0, 1., 0, 0]))

    assert np.all(fetal_genotype_prior('0|1', '0/0', '0-0') == np.array([1., 0, 0, 0]))
    assert np.all(fetal_genotype_prior('0|1', '0/1', '0-0') == np.array([0.5, 0.5, 0, 0]))
    assert np.all(fetal_genotype_prior('0|1', '1/1', '0-0') == np.array([0, 1., 0, 0]))

    assert np.all(fetal_genotype_prior('1|1', '0/0', '0-0') == np.array([0, 0, 1., 0]))
    assert np.all(fetal_genotype_prior('1|1', '0/1', '0-0') == np.array([0, 0, 0.5, 0.5]))
    assert np.all(fetal_genotype_prior('1|1', '1/1', '0-0') == np.array([0, 0, 0, 1.]))

    assert np.all(fetal_genotype_prior('0|0', '0/0', '1-0') == np.array([1., 0, 0, 0]))
    assert np.all(fetal_genotype_prior('0|0', '0/1', '1-0') == np.array([0.5, 0.5, 0, 0]))
    assert np.all(fetal_genotype_prior('0|0', '1/1', '1-0') == np.array([0, 1., 0, 0]))

    assert np.all(fetal_genotype_prior('0|1', '0/0', '1-0') == np.array([0, 0, 1., 0]))
    assert np.all(fetal_genotype_prior('0|1', '0/1', '1-0') == np.array([0, 0, 0.5, 0.5]))
    assert np.all(fetal_genotype_prior('0|1', '1/1', '1-0') == np.array([0, 0, 0, 1.]))

    assert np.all(fetal_genotype_prior('1|1', '0/0', '1-0') == np.array([0, 0, 1., 0]))
    assert np.all(fetal_genotype_prior('1|1', '0/1', '1-0') == np.array([0, 0, 0.5, 0.5]))
    assert np.all(fetal_genotype_prior('1|1', '1/1', '1-0') == np.array([0, 0, 0, 1.]))

    # phased dad
    assert np.all(fetal_genotype_prior('0/0', '0|0', '0-0') == np.array([1., 0, 0, 0]))
    assert np.all(fetal_genotype_prior('0/0', '0|1', '0-0') == np.array([1., 0, 0, 0]))
    assert np.all(fetal_genotype_prior('0/0', '1|1', '0-0') == np.array([0, 1., 0, 0]))

    assert np.all(fetal_genotype_prior('0/1', '0|0', '0-0') == np.array([0.5, 0, 0.5, 0]))
    assert np.all(fetal_genotype_prior('0/1', '0|1', '0-0') == np.array([0.5, 0, 0.5, 0]))
    assert np.all(fetal_genotype_prior('0/1', '1|1', '0-0') == np.array([0, 0.5, 0, 0.5]))

    assert np.all(fetal_genotype_prior('1/1', '0|0', '0-0') == np.array([0, 0, 1., 0]))
    assert np.all(fetal_genotype_prior('1/1', '0|1', '0-0') == np.array([0, 0, 1., 0]))
    assert np.all(fetal_genotype_prior('1/1', '1|1', '0-0') == np.array([0, 0, 0, 1.]))

    assert np.all(fetal_genotype_prior('0/0', '0|0', '0-1') == np.array([1., 0, 0, 0]))
    assert np.all(fetal_genotype_prior('0/0', '0|1', '0-1') == np.array([0, 1., 0, 0]))
    assert np.all(fetal_genotype_prior('0/0', '1|1', '0-1') == np.array([0, 1., 0, 0]))

    assert np.all(fetal_genotype_prior('0/1', '0|0', '0-1') == np.array([0.5, 0, 0.5, 0]))
    assert np.all(fetal_genotype_prior('0/1', '0|1', '0-1') == np.array([0, 0.5, 0, 0.5]))
    assert np.all(fetal_genotype_prior('0/1', '1|1', '0-1') == np.array([0, 0.5, 0, 0.5]))

    assert np.all(fetal_genotype_prior('1/1', '0|0', '0-1') == np.array([0, 0, 1., 0]))
    assert np.all(fetal_genotype_prior('1/1', '0|1', '0-1') == np.array([0, 0, 0, 1.]))
    assert np.all(fetal_genotype_prior('1/1', '1|1', '0-1') == np.array([0, 0, 0, 1.]))

    # unphased genotype
    assert np.all(fetal_genotype_prior('0/0', '0/0', '0-0') == np.array([1., 0, 0, 0]))
    assert np.all(fetal_genotype_prior('0/0', '0/1', '0-0') == np.array([0.5, 0.5, 0, 0]))
    assert np.all(fetal_genotype_prior('0/0', '1/1', '0-0') == np.array([0, 1., 0, 0]))

    assert np.all(fetal_genotype_prior('0/1', '0/0', '0-0') == np.array([0.5, 0, 0.5, 0]))
    assert np.all(fetal_genotype_prior('0/1', '0/1', '0-0') == np.array([0.25, 0.25, 0.25, 0.25]))
    assert np.all(fetal_genotype_prior('0/1', '1/1', '0-0') == np.array([0, 0.5, 0, 0.5]))

    assert np.all(fetal_genotype_prior('1/1', '0/0', '0-0') == np.array([0, 0, 1., 0]))
    assert np.all(fetal_genotype_prior('1/1', '0/1', '0-0') == np.array([0, 0, 0.5, 0.5]))
    assert np.all(fetal_genotype_prior('1/1', '1/1', '0-0') == np.array([0, 0, 0, 1.]))


    ## transition probabilities
    out = haplotype_transition_probability(
        previous_allele_origin = '0-0',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.6724, 0.1476, 0.1476, 0.0324]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '0-1',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.1476, 0.6724, 0.0324, 0.1476]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '1-0',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.1476, 0.0324, 0.6724, 0.1476]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '1-1',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.0324, 0.1476, 0.1476, 0.6724]),
                               rtol=1e-5, atol=0)

    ## transition probabilities (first locus)
    out = haplotype_transition_probability(
        previous_allele_origin = None,
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.25, 0.25, 0.25, 0.25]),
                               rtol=1e-5, atol=0)

    ## transition probabilities (unphased locus for mother)
    out = haplotype_transition_probability(
        previous_allele_origin = '0-0',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = False, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.41, 0.09, 0.41, 0.09]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '0-1',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = False, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.09, 0.41, 0.09, 0.41]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '1-0',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = False, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.41, 0.09, 0.41, 0.09]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '1-1',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = False, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.09, 0.41, 0.09, 0.41]),
                               rtol=1e-5, atol=0)

    ## transition probabilities (unphased locus for father)
    out = haplotype_transition_probability(
        previous_allele_origin = '0-0',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = False)
    np.testing.assert_allclose(out, np.array([0.41, 0.41, 0.09, 0.09]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '0-1',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = False)
    np.testing.assert_allclose(out, np.array([0.41, 0.41, 0.09, 0.09]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '1-0',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = False)
    np.testing.assert_allclose(out, np.array([0.09, 0.09, 0.41, 0.41]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '1-1',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = False)
    np.testing.assert_allclose(out, np.array([0.09, 0.09, 0.41, 0.41]),
                               rtol=1e-5, atol=0)

    ## transition probabilities (missing single allele origin for mother)
    out = haplotype_transition_probability(
        previous_allele_origin = '-0',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.41, 0.09, 0.41, 0.09]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '-1',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.09, 0.41, 0.09, 0.41]),
                               rtol=1e-5, atol=0)

    ## transition probabilities (missing single allele origin for father)
    out = haplotype_transition_probability(
        previous_allele_origin = '0-',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.41, 0.41, 0.09, 0.09]),
                               rtol=1e-5, atol=0)

    out = haplotype_transition_probability(
        previous_allele_origin = '1-',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.09, 0.09, 0.41, 0.41]),
                               rtol=1e-5, atol=0)

    ## transition probabilities (missing both allele origins)
    out = haplotype_transition_probability(
        previous_allele_origin = '-',
        mother_phasing_error_proba = 0.1, father_phasing_error_proba = 0.1,
        locus_dist = 10, recombination_rate = 0.01,
        mother_phased = True, father_phased = True)
    np.testing.assert_allclose(out, np.array([0.25, 0.25, 0.25, 0.25]),
                               rtol=1e-5, atol=0)

    ## single SNP conditional posterior
    possible_gt = [] # ['0/0', '0/1', '1/1']
    possible_hap = ['0|0', '0|1', '1|0', '1|1']
    possible_allele_origin = ['0-0', '0-1', '1-0', '1-1']
    ff = 0.2
    coverage = 100
    add_noise = False
    verbose = False
    tol = 0.05

    for mother_hap, father_hap, previous_allele_origin \
        in itertools.product(possible_hap + possible_gt,
                             possible_hap + possible_gt,
                             possible_allele_origin):
        print("--------------------------------------------------------------")
        fetal_gt, cfdna_gt, cfdna_ad = simulation.single_snp_data(
                                        mother_hap, father_hap, previous_allele_origin,
                                        ff, coverage, add_noise, verbose)
        print("maternal haplotype = {} -- paternal haplotype = {}"
                .format(mother_hap, father_hap))
        print("allele origin = {}".format(previous_allele_origin))

        # fetal fraction estimation
        n_read = np.sum(cfdna_ad)
        estim_ff = estimate_local_fetal_fraction(
                            mother_hap, father_hap,
                            cfdna_gt, cfdna_ad, n_read, tol)

        if estim_ff is None:
            print("estimated ff = {} -- true ff = {:.4f}".format(estim_ff, ff))
            estim_ff = ff
        else:
            print("estimated ff = {:.4f} -- true ff = {:.4f}"
                  .format(estim_ff, ff))

        # data loglikelihood
        data_likelihood = read_data_likelihood(mother_hap, father_hap, cfdna_gt,
                                               cfdna_ad, estim_ff)
        print("data likelihood = {}".format(data_likelihood))
        # transition probability
        trans_prob = haplotype_transition_probability(
                                previous_allele_origin,
                                mother_phasing_error_proba = 1e-10,
                                father_phasing_error_proba = 1e-10,
                                locus_dist = 2000, recombination_rate = 1.2e-8,
                                mother_phased = True,
                                father_phased = True)
        print("transition probabilities = {}".format(trans_prob))

        # posterior
        cond_posteriors = model_conditional_posterior(
                                mother_hap, father_hap, cfdna_gt, cfdna_ad, ff,
                                previous_allele_origin,
                                mother_phasing_error_proba = 1e-10,
                                father_phasing_error_proba = 1e-10,
                                locus_dist = 2000, recombination_rate = 1.2e-8,
                                mother_phased = True,
                                father_phased = True)
        print("conditional posterior = {}".format(cond_posteriors))

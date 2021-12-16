#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
import numpy as np
from scipy.spatial.distance import cdist
import warnings
# internal
from prediag.filter import snp_check
from prediag.utils import find_ad, format_input, parse_gt, round0, softmax, unparse_gt


def fetal_genotype_prior(mother_gt, father_gt):
    """Compute fetal genotype prior according to Mendelian law

    Maternal and paternal genotype: '0/0', '0/1', '1/1'
    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele

    Input:
        mother_gt (list): maternal genotype ['x', 'y'] with x,y in {0,1}.
        father_gt (list): paternal genotype ['x', 'y'] with x,y in {0,1}.

    Output: vector of prior probability for each fetal genotype 0/0, 0/1, 1/0,
    1/1 (in this order).
    """
    out = None

    if isinstance(mother_gt, str):
        mother_gt = parse_gt(mother_gt)
    if isinstance(father_gt, str):
        father_gt = parse_gt(father_gt)

    if '2' in mother_gt or '2' in father_gt:
        warnings.warn("Poly-allelic SNPs (i.e. with alleles 0,1,2) are not "
                      + "accounted for at the moment.")
        return out

    # possible fetal genotypes
    possible_fetal_gt = np.array(['0/0', '0/1', '1/0', '1/1'])
    # fetal gt
    fetal_gt = np.unique(['{}/{}'.format(mat, pat) for mat in mother_gt
                                                    for pat in father_gt])
    # update fetal gt prior
    out = np.array([0., 0., 0., 0.])
    out[np.where(np.isin(possible_fetal_gt, fetal_gt))] = 1./len(fetal_gt)
    # output
    return out


def read_data_loglikelihood(mother_gt, father_gt, cfdna_gt, cfdna_ad, ff,
                            single = False, read_val = 0):
    """Compute data (reads) log-likelihood knowing fetal genotype for a SNP

    Maternal and paternal genotype: '0/0', '0/1', '1/1'
    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele

    Potential input for genotypes:
        - ['x', 'y'] with x, y in {0,1}.
        - 'x/y' or 'x|y' with x, y in {0,1}.

    Potential input for allelic depth: np.array or list.

    Input:
        mother_gt: maternal genotype.
        father_gt: paternal genotype.
        cfdna_gt: plasma genotype.
        cfdna_ad: vector of allele depth in plasma (read count per allele).
        ff (float): fetal fraction between 0 and 1.
        single (boolean): if True, single read data log-likelihood is returned
            corresponding to read allele value given by input parameter
            `read_val`. If False, full data log-likelihood for all reads
            covering a locus is returned. Default is False.
        read_val (int): read allele in 0, 1, only used with single read data
            log-likelihood (i.e. with input parameter `single` = True).
            Default value is 0.

    Output: vector of data log-probability knowing each fetal genotype 0/0,
    0/1, 1/0, 1/1 (in this order).
    """
    out = None

    mother_gt, father_gt, cfdna_gt, cfdna_ad = format_input(
            mother_gt, father_gt, cfdna_gt, cfdna_ad
    )

    # log p(data | fetal_gt)
    #    = \sum_j log p(read_j | fetal_gt, mother_gt, ff)
    #    = \sum_k #{ read =k } * log p(read = k | fetal_gt, mother_gt, ff)
    # log p(read = k | fetal_gt, mother_gt, ff)
    #    = log( p(read = k | fet) * ff + p(read = k | mat) * (1 - ff) )

    if snp_check(mother_gt, father_gt, cfdna_gt, cfdna_ad):
        ## number of reads per allele
        N_0 = 0
        N_1 = 0
        # all read mode (default)
        if not single:
            N_0 = find_ad('0', cfdna_gt, cfdna_ad)
            N_1 = find_ad('1', cfdna_gt, cfdna_ad)
        # single read mode
        else:
            N_0 = int(read_val == 0)
            N_1 = int(read_val == 1)

        ## mother allele number
        mother_n0 = list(mother_gt).count('0')
        mother_n1 = list(mother_gt).count('1')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## knowing fetal_gt = 0/0 (mother_gt = 1/1 not possible)
            # log p(read = 0 | fetal_gt, mother_gt, ff)
            logpr0 = None
            if N_0 > 0:
                logpr0 = N_0 * np.log(1. * ff + mother_n0/2 * (1-ff),
                                      dtype = np.float64) \
                            if mother_n0 > 0 \
                            else np.log(0, dtype = np.float64)
            else:
                logpr0 = 0.
            # log p(read = 1 | fetal_gt, mother_gt, ff)
            logpr1 = None
            if N_1 > 0:
                logpr1 = N_1 * np.log(0. * ff + mother_n1/2 * (1-ff),
                                      dtype = np.float64) \
                            if mother_n0 > 0 \
                            else np.log(0, dtype = np.float64)
            else:
                logpr1 = 0.
            # finally
            logp00 = logpr0 + logpr1
            ## knowing fetal_gt = 0/1 (alt from father, mother_gt = 1/1 not possible)
            # log p(read = 0 | fetal_gt, mother_gt, ff)
            logpr0 = None
            if N_0 > 0:
                logpr0 = N_0 * np.log(0.5 * ff + mother_n0/2 * (1-ff),
                                      dtype = np.float64) \
                            if mother_n0 > 0 \
                            else np.log(0, dtype = np.float64)
            else:
                logpr0 = 0.
            # log p(read = 1 | fetal_gt, mother_gt, ff)
            logpr1 = None
            if N_1 > 0:
                logpr1 = N_1 * np.log(0.5 * ff + mother_n1/2 * (1-ff),
                                      dtype = np.float64) \
                            if mother_n0 > 0 \
                            else np.log(0, dtype = np.float64)
            else:
                logpr1 = 0.
            # finally
            logp01 = logpr0 + logpr1
            ## knowing fetal_gt = 1/0 (alt from the mother, mother_gt = 0/0 not possible)
            # log p(read = 0 | fetal_gt, mother_gt, ff)
            logpr0 = None
            if N_0 > 0:
                logpr0 = N_0 * np.log(0.5 * ff + mother_n0/2 * (1-ff),
                                      dtype = np.float64) \
                            if mother_n1 > 0 \
                            else np.log(0, dtype = np.float64)
            else:
                logpr0 = 0.
            # log p(read = 1 | fetal_gt, mother_gt, ff)
            logpr1 = None
            if N_1 > 0:
                logpr1 = N_1 * np.log(0.5 * ff + mother_n1/2 * (1-ff),
                                      dtype = np.float64) \
                            if mother_n1 > 0 \
                            else np.log(0, dtype = np.float64)
            else:
                logpr1 = 0.
            # finally
            logp10 = logpr0 + logpr1
            ## knowing fetal_gt = 1/1 (mother_gt = 0/0 not possible)
            # log p(read = 0 | fetal_gt, mother_gt, ff)
            logpr0 = None
            if N_0 > 0:
                logpr0 = N_0 * np.log(0. * ff + mother_n0/2 * (1-ff),
                                      dtype = np.float64) \
                            if mother_n1 > 0 \
                            else np.log(0, dtype = np.float64)
            else:
                logpr0 = 0.
            # log p(read = 1 | fetal_gt, mother_gt, ff)
            logpr1 = None
            if N_1 > 0:
                logpr1 = N_1 * np.log(1. * ff + mother_n1/2 * (1-ff),
                                      dtype = np.float64) \
                            if mother_n1 > 0 \
                            else np.log(0, dtype = np.float64)
            else:
                logpr1 = 0.
            # finally
            logp11 = logpr0 + logpr1
            ## Output
            out = np.array([logp00, logp01, logp10, logp11])
    else:
        out = np.log([0.25, 0.25, 0.25, 0.25], dtype = np.float64)

    return out


def model_posterior(data_loglikelihood, fetal_gt_prior, tol = 0.001,
                    return_log = False):
    """Compute fetal genotype posterior knowing the data (reads) for a SNP

    Maternal and paternal genotype: '0/0', '0/1', '1/1'
    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele

    Input:
        data_loglikelihood (np.array): vector of data log-likelihood knowing
            each fetal genotype 0/0, 0/1, 1/0, 1/1 (in this order).
        fetal_gt_prior (np.array): vector of prior probabilities for each
            fetal genotype 0/0, 0/1, 1/0, 1/1 (in this order).
        tol (float): tolerance for number comparison to zero. Default is 1e-3.
        return_log (boolean): if True, return un-normalized log-posteriors
            (i.e. joint log-likelihood), else return posterior probabilities.
            Default is False.

    Output:
        prediction (string): predicted fetal genotype for the considered SNP
            by Maximum A Posteriori (MAP).
        posteriors (np.array): vector of posterior probability for each
            fetal genotype 0/0, 0/1, 1/0, 1/1 (in this order).
    """
    out = ('', None)
    genotype_candidates = np.array(['0/0', '0/1', '1/0', '1/1'])

    if fetal_gt_prior is not None and data_loglikelihood is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## fetal genotype log-priors
            fetal_gt_logprior = np.log(fetal_gt_prior, dtype = np.float64)
            ## joint log-likelihood = log p(data | fetal_genotype) + log p(fetal_genotype)
            joint_loglikelihood = np.add(data_loglikelihood, fetal_gt_logprior,
                                         dtype = np.float64)
            ## genotype prediction by Maximum A Posteriori (MAP)
            prediction = genotype_candidates[joint_loglikelihood.argmax()]
            ## Compute posterior probabilities with safeguards regarding NaN production
            # in exponential
            posteriors = round0(softmax(joint_loglikelihood))
            ## check for equality in prediction
            valid_dist = cdist(np.array(posteriors.max()).reshape(1,1),
                               np.delete(posteriors,
                                         posteriors.argmax()).reshape(-1, 1),
                               'minkowski', p=1.)
            if np.any(valid_dist < tol):
                prediction = None
            ## return un-normalized log posteriors (i.e. joint log-likelihood)
            if return_log:
                posteriors = joint_loglikelihood
            ## output
            out = (prediction, posteriors)

    return out


# example
if __name__ == '__main__':
    import itertools
    from prediag.fetal_fraction import estimate_local_fetal_fraction
    import prediag.simulation as simulation

    ## priors
    assert np.all(fetal_genotype_prior('0/0', '0/0') == np.array([1., 0, 0, 0]))
    assert np.all(fetal_genotype_prior('0/0', '0/1') == np.array([0.5, 0.5, 0, 0]))
    assert np.all(fetal_genotype_prior('0/0', '1/1') == np.array([0, 1., 0, 0]))

    assert np.all(fetal_genotype_prior('0/1', '0/0') == np.array([0.5, 0, 0.5, 0]))
    assert np.all(fetal_genotype_prior('0/1', '0/1') == np.array([0.25, 0.25, 0.25, 0.25]))
    assert np.all(fetal_genotype_prior('0/1', '1/1') == np.array([0, 0.5, 0, 0.5]))

    assert np.all(fetal_genotype_prior('1/1', '0/0') == np.array([0, 0, 1., 0]))
    assert np.all(fetal_genotype_prior('1/1', '0/1') == np.array([0, 0, 0.5, 0.5]))
    assert np.all(fetal_genotype_prior('1/1', '1/1') == np.array([0, 0, 0, 1.]))

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
                                        ff, coverage, add_noise, verbose)

        # fetal fraction estimation
        n_read = np.sum(cfdna_ad)
        estim_ff = estimate_local_fetal_fraction(
                            mother_gt, father_gt,
                            cfdna_gt, cfdna_ad, n_read, tol)

        if estim_ff is None:
            print("estimated ff = {} -- true ff = {:.4f}".format(estim_ff, ff))
            estim_ff = ff
        else:
            print("estimated ff = {:.4f} -- true ff = {:.4f}"
                  .format(estim_ff, ff))

        # fetal genotype priors
        fetal_gt_prior = fetal_genotype_prior(mother_gt, father_gt)
        print("fetal genotype priors = {}".format(fetal_gt_prior))
        # data loglikelihood
        data_loglikelihood = read_data_loglikelihood(mother_gt, father_gt, cfdna_gt,
                                                     cfdna_ad, estim_ff)
        print("data loglikelihood = {}".format(data_loglikelihood))
        # posterior
        prediction, posteriors = model_posterior(data_loglikelihood, fetal_gt_prior,
                                                 tol = 0.001)
        print("posterior = {}".format(posteriors))
        print("predicted fetal genotype = {}".format(prediction))

#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
import datetime
from joblib import delayed, Parallel
import psutil
import numpy as np
import pandas as pds
import sys
from textwrap import dedent
import time
from tqdm import tqdm
# internal
from prediag.hap_model import model_conditional_posterior
from prediag.parallel import ProgressParallel
from prediag.utils import float2string, index_allele_origin, is_het, is_phased, parse_allele_origin, parse_gt, readable_allele_origin
from prediag.vcf_reader import parse_region


def init(seq_data_tab, init_allele_origin_tab, both_parent_phased = True):
    """Compute intial values for Gibbs sampler

    Input:
        seq_data_tab (Pandas.DataFrame): sequencing data table produced
            by 'prediag.vcf_reader.load_vcf_data' function.
        init_allele_origin_tab (Pandas.DataFrame): initialization tab
            produced by `prediag.heuristic_fetal_allele_origin.infer_parental_allele_origin`
            function.
        both_parent_phased (bool): indicates if keeping loci only where both
            parents are phased (True) or where a single parent is phased
            (False). Default is True.
    """
    # prepare output
    out = init_allele_origin_tab.copy()
    # convert allele origin to index if human readable version
    try:
        out['allele_origin'] = out['allele_origin'].apply(index_allele_origin)
    except:
        pass
    # remove loci where no parents are phased
    if both_parent_phased:
        out = out[np.logical_and(out['mother_gt'].apply(is_phased),
                                out['father_gt'].apply(is_phased))]
    else:
        out = out[np.logical_or(out['mother_gt'].apply(is_phased),
                                out['father_gt'].apply(is_phased))]
    # drop useless columns
    out.drop(['fetal_gt_pred', 'fetal_gt_posterior'], inplace=True, axis=1)
    # add missing columns from input
    out = pds.merge(out.dropna(),
                    seq_data_tab[['chrom', 'pos', 'mother_pq', 'mother_jq',
                                  'father_pq', 'father_jq']],
                    how='left', on=['chrom', 'pos'])
    out['allele_origin_init'] = out['allele_origin']
    # output
    return out


def sampler(loci_tab, recombination_rate = 1.2e-8, verbose = False):
    """Sampler function used in Gibbs sampler iterations

    Input:
        loci_tab (Pandas.DataFrame): table of loci with following fields
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
            * mother_pq (float): mother phasing quality probability, "probability
                that alleles are phased incorrectly in a heterozygous call"
                (10x-genomics doc).
            * mother_jq (float): mother junction quality probability, "probability
                that there is a large-scale phasing switch error occuring between
                this variant and the following variant" (10x-genomics doc).
            * father_pq (float): father phasing quality probability, "probability
                that alleles are phased incorrectly in a heterozygous call"
                (10x-genomics doc).
            * father_jq (float): father junction quality probability, "probability
                that there is a large-scale phasing switch error occuring between
                this variant and the following variant" (10x-genomics doc).
            * fetal_fraction (float): estimated fetal fraction.
            * allele_origin (string): fetal allele origin in parental haplotypes,
                i.e. 'a-b' with a, b in {0,1}.
            * allele_origin_conf (float list list): vector of confidence
                probabilities for each allele origin.
        recombination_rate (float): recombination rate per bp (between 0 and 1).
            Default value 1.2e-8.
        verbose (bool): set verbosity.

    Output: updated `loci_tab` table.
    """
    possible_allele_origin = ['0-0', '0-1', '1-0', '1-1']
    previous_allele_origin = None
    previous_pos = 0
    previous_chrom = ""
    # iterate through sequencing data table and estimate fetal fraction
    for index, row in loci_tab.iterrows():
        # current locus info
        current_chrom = row["chrom"]
        current_pos = row["pos"]
        mother_gt = row["mother_gt"]
        father_gt = row["father_gt"]
        cfdna_gt = row["cfdna_gt"]
        cfdna_ad = row["cfdna_ad"]
        ff = row["fetal_fraction"]
        mother_pq = row["mother_pq"] if is_het(parse_gt(mother_gt)) else 0
        father_pq = row["father_pq"] if is_het(parse_gt(father_gt)) else 0

        # check if chromsome switch
        if current_chrom != previous_chrom:
            previous_allele_origin = None

        # conditional posterior
        cond_posteriors = model_conditional_posterior(
            mother_gt, father_gt, cfdna_gt, cfdna_ad, ff,
            previous_allele_origin,
            mother_phasing_error_proba = mother_pq,
            father_phasing_error_proba = father_pq,
            locus_dist = np.abs(current_pos - previous_pos),
            recombination_rate = recombination_rate,
            mother_phased = is_phased(mother_gt),
            father_phased = is_phased(father_gt),
            verbose = verbose
        )

        # sample from conditional posterior
        current_allele_origin = np.random.choice(
            possible_allele_origin, size=None, p=cond_posteriors
        )
        # next locus
        previous_pos = current_pos
        previous_chrom = current_chrom
        previous_allele_origin = current_allele_origin
        # update table
        loci_tab.at[index, 'allele_origin'] = current_allele_origin
        loci_tab.at[index, 'allele_origin_conf'] = list(cond_posteriors)
    # output
    return loci_tab


def gibbs_sampling(init_tab, n_burn = 200, n_iter = 500, lag = 10,
                   recombination_rate = 1.2e-8, verbose = False,
                   position = 0, total = 1, **kwargs):
    """Gibbs sampler

    Input:
        init_tab (Pandas.DataFrame): table returned by `init` function.
        n_burn (int): number of burning iterations.
        n_iter (int): number of sampling iterations (after burning period).
        lag (int): actually keep a sample every `lag` iterations.
        recombination_rate (float): recombination rate per bp (between 0 and 1).
            Default value 1.2e-8.
        verbose (bool): verbosity. Default is False.
        position (int): position for parallel Gibbs sampling.
        total (int): total number of samplers for parallel Gibbs sampling.
    """
    # initialization
    loci_tab = init_tab.copy()
    # burning iterations
    text = 'Gibbs burn'
    if total > 1:
        text = 'Gibbs burn #{}'.format(position)
    else:
        text = 'Gibbs burn'
    for index in tqdm(range(int(n_burn)), position = int(position),
                      desc = text, mininterval = 1, disable = not verbose):
        loci_tab = sampler(loci_tab, recombination_rate, verbose = False)
    # short break to avoid issue with progress bars
    time.sleep(2)
    sys.stdout.flush()
    sys.stderr.flush()
    # sampling iterations
    out = []
    text = 'Gibbs samp'
    if total > 1:
        text = 'Gibbs samp #{}'.format(position)
    else:
        text = 'Gibbs samp'
    for index in tqdm(range(int(n_iter)), position = int(position + total),
                      desc = text, mininterval = 1, disable = not verbose):
        loci_tab = sampler(loci_tab, recombination_rate, verbose = False)
        if index % lag == 0:
            out.append(loci_tab['allele_origin'].copy().to_numpy())
    if verbose:
        print("")
    sys.stdout.flush()
    sys.stderr.flush()
    # format output
    sample_tab = pds.DataFrame(out,
                               columns = loci_tab['chrom'] + '-'
                                            + loci_tab['pos'].apply(str))
    # output
    return sample_tab


def parallel_gibbs_sampling(init_tab, n_burn = 100, n_iter = 500,
                            lag = 50, n_thread = 0, n_gibbs = 20,
                            recombination_rate = 1.2e-8, verbose = False,
                            **kwargs):
    """Parralel Gibbs sampler

    Input:
        init_tab (Pandas.DataFrame): table returned by `init` function.
        n_burn (int): number of burning iterations.
        n_iter (int): number of sampling iterations (after burning period).
        lag (int): actually keep a sample every `lag` iterations.
        n_thread (int): number of threads for parallel computing, if 0 then
            all cpu cores are used.
        n_gibbs (int): number of parallel Gibbs samplers.
        recombination_rate (float): recombination rate per bp (between 0 and 1).
            Default value 1.2e-8.
        verbose (bool): verbosity. Default is False.
    """

    # number of threads
    if n_thread == 0:
        n_thread = psutil.cpu_count(logical=False)
    # parralel Gibbs sampler indexes
    inputs = np.arange(n_gibbs)
    # run
    processed_list = Parallel(n_jobs=n_thread)(
                            delayed(gibbs_sampling)
                            (init_tab, n_burn, n_iter, lag, recombination_rate,
                             verbose, i, n_gibbs) for i in inputs)
    # output
    return pds.concat(processed_list, ignore_index = True)


def extract_posterior(sample_tab):
    """Extract posterior parametrization from output of Gibbs sampling

    Fetal allele origin in parental haplotypes:
        - 'mat1-pat1', 'mat1-pat2', 'mat2-pat1', 'mat2-pat2' (lexicographic order)
        - with following convention: 'matA-patB' where A = maternal haplotype
        origin, and B = paternal haplotype origin, with 1 = haplotype 1 and
        2 = haplotype 2, i.e. if parental haplotype = "x|y", 1 corresponds to x,
        and 2 to y.
        - index version: 'a-b' with a, b in {0, 1}, a = index of maternal
        haplotype origin at the locus, b = index of paternal haplotype
        origin at the locus, 0 corresponds to haplotype 1, and
        1 corresponds to haplotype 2

    Objective:
        * find if allele 'A' is 'x' (index 0) or 'y' (index 1) in maternal
          haplotype 'x|y'.
        * find if allele 'B' is 'x' (index 0) or 'y' (index 1) in paternal
          haplotype 'x|y'.

    Input:
        sample_tab (Pandas.DataFrame): output samples from Gibbs sampler,
            loci in columns and fetal allele origin for each sample in rows.
    """
    # marginal posterior
    possible_allele_origin = ['0-0', '0-1', '1-0', '1-1']
    post_tab = sample_tab.apply(
                lambda x: x.value_counts(sort = False)).reindex(
                    possible_allele_origin).fillna(0).apply(
                        lambda x: x/np.sum(x)).T
    # output
    return post_tab


def infer_origin(loci_tab, post_tab, parent = "mother"):
    """Infer allele origin of the region regarding the parental allele specified
    in input

    Fetal allele origin in parental haplotypes:
        - index version: 'a-b' with a, b in {0, 1}, a = index of maternal
        haplotype origin at the locus, b = index of paternal haplotype
        origin at the locus, 0 corresponds to haplotype 1, and
        1 corresponds to haplotype 2

    Objective:
        * find if allele 'A' is 'x' (index 0) or 'y' (index 1) in maternal
          haplotype 'x|y'.
        * find if allele 'B' is 'x' (index 0) or 'y' (index 1) in paternal
          haplotype 'x|y'.

    Input:
        loci_tab (Pandas.DataFrame): output of `sampler` function.
        post_tab (Pandas.DataFrame): output of the `extract_posterior` function,
            loci in rows and parental allele origin combination in columns.
        parent (string): 'mother' or 'father'.
    """
    # parental allele origin
    parental_allele = np.array(list(map(lambda x : parse_allele_origin(x),
                                        post_tab.columns)))
    mat_allele = parental_allele[:,0]
    pat_allele = parental_allele[:,1]
    # choose a parent
    tag = None
    if parent == "mother":
        parental_allele = mat_allele
        tag = 'mat'
    elif parent == "father":
        parental_allele = pat_allele
        tag = 'pat'
    else:
        raise ValueError("`parent` input parameter should be 'mother' or 'father'.")
    # find index of haplotypes 0 and 1
    hap0_id = np.where(parental_allele == 0)[0]
    hap1_id = np.where(parental_allele == 1)[0]
    # prepare output
    out = post_tab.copy()
    # compute marginal probabilities
    out[tag + '1_hap_post'] = out[out.columns[list(hap0_id)]].sum(axis=1)
    out[tag + '2_hap_post'] = out[out.columns[list(hap1_id)]].sum(axis=1)
    # add loci location
    out['chrom'] = list(map(lambda x: parse_region(x)[0],
                            out.index))
    out['pos'] = list(map(lambda x: parse_region(x)[1],
                          out.index))
    # discard useless columns
    out = out[['chrom', 'pos', tag + '1_hap_post', tag + '2_hap_post']]
    # output
    return out


def print_region_origin(res_tab, parent = "mother", threshold = 0.65):
    """Print allele origin inference summary information for the region

    Input:
        res_tab (Pandas.DataFrame): output of `infer_origin` function.
        parent (string): 'mother' or 'father'.
        threshold (float): voting threshold for probabilities. Default value
            is 0.65 (65%).
    """
    # choose a parent
    tag = None
    long_tag = None
    if parent == "mother":
        tag = 'mat'
        long_tag = 'maternal'
    elif parent == "father":
        tag = 'pat'
        long_tag = 'paternal'
    else:
        raise ValueError("`parent` input parameter should be 'mother' or 'father'.")
    # count locus
    n_loci = int(len(res_tab.index))
    n_1 = int(np.sum(res_tab[tag + '1_hap_post'] > threshold))
    n_2 = int(np.sum(res_tab[tag + '2_hap_post'] > threshold))
    n_indecise = n_loci - n_1 - n_2
    # output
    text = dedent('''
        {} allele origin in fetus:
            hap. 1 = {}% ({}/{} loci)
            hap. 2 = {}% ({}/{} loci)
            indecise = {}% ({}/{} loci)'''
        .format(
            long_tag,
            float2string(100 * n_1 / n_loci), n_1, n_loci,
            float2string(100 * n_2 / n_loci), n_2, n_loci,
            float2string(100 * n_indecise / n_loci), n_indecise, n_loci))
    # print
    print(text)


def infer_parental_allele_origin(
    seq_data_tab, init_allele_origin_tab, n_burn = 100, n_iter = 500,
    lag = 50, n_thread = 0, n_gibbs = 20, recombination_rate = 1.2e-8,
    both_parent_phased = True, verbose = False, filename = None, **kwargs
):
    """Bayesian inference of fetal allele origin among parental (phased)
    haplotypes, on a whole region.

    Maternal and paternal genotype: '0|0', '0|1', '1|0', '1|1'
    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele

    Fetal allele origin in parental haplotypes:
        - 'mat1-pat1', 'mat1-pat2', 'mat2-pat1', 'mat2-pat2' (lexicographic order)
        - with following convention: 'matA-patB' where A = maternal haplotype
        origin, and B = paternal haplotype origin, with 1 = haplotype 1 and
        2 = haplotype 2, i.e. if parental haplotype = "x|y", 1 corresponds to x,
        and 2 to y.
        - index version: 'a-b' with a, b in {0, 1}, a = index of maternal
        haplotype origin at the locus, b = index of paternal haplotype
        origin at the locus, 0 corresponds to haplotype 1, and
        1 corresponds to haplotype 2

    Objective:
        * find if allele 'A' is 'x' (index 0) or 'y' (index 1) in maternal
          haplotype 'x|y'.
        * find if allele 'B' is 'x' (index 0) or 'y' (index 1) in paternal
          haplotype 'x|y'.

    Input:
        seq_data_tab (Pandas.DataFrame): sequencing data table produced
            by 'prediag.vcf_reader.load_vcf_data' function.
        init_allele_origin_tab (Pandas.DataFrame): initialization tab
            produced by `prediag.heuristic_fetal_allele_origin.infer_parental_allele_origin`
            function.
        n_burn (int): number of burning iterations.
        n_iter (int): number of sampling iterations (after burning period).
        lag (int): actually keep a sample every `lag` iterations.
        n_thread (int): number of threads for parallel computing, if 0 then
            all cpu cores are used.
        n_gibbs (int): number of parallel Gibbs samplers.
        recombination_rate (float): recombination rate per bp (between 0 and 1).
            Default value 1.2e-8.
        both_parent_phased (bool): indicates if keeping loci only where both
            parents are phased (True) or where a single parent is phased
            (False). Default is True.
        verbose (bool): verbosity. Default is False.
        filename (string): file to store result per locus table.

    Output: Pandas.DataFrame with for each SNP
        * chrom (string): chromosome
        * pos (integer): locus position on the sequence.
        * mother_gt (string): maternal haplotype 'x|y' with x, y in {0,1},
            or maternal genotype 'x/y' if haplotype not available.
        * father_gt (string): paternal haplotype 'x|y' with x, y in {0,1},
            or paternal genotype 'x/y' if haplotype not available.
        * cfdna_gt (string): cfdna genotype 'x/y' with x, y in {0,1}.
        * cfdna_ad (int list): allelic depth at the locus.
        * cfdna_dp (int): coverage at the locus.
        * fetal_fraction (float): smoothed estimated fetal fraction at the locus.
        * mat1_hap_post (float): posterior probability that fetal maternal
            allele originates from maternal haplotype 1.
        * mat2_hap_post (float): posterior probability that fetal maternal
            allele originates from maternal haplotype 2.
        * pat1_hap_post (float): posterior probability that fetal paternal
            allele originates from paternal haplotype 1.
        * pat2_hap_post (float): posterior probability that fetal paternal
            allele originates from paternal haplotype 2.
    """
    # initialization
    loci_tab = init(seq_data_tab, init_allele_origin_tab, both_parent_phased)
    # context
    if verbose:
        print("Dimension of the region (nb of SNPs) = {}"
              .format(len(loci_tab.index)))
    # run
    sample_tab = None
    t0 = time.time()
    if n_thread == 1:
        sample_tab = gibbs_sampling(loci_tab, n_burn, n_iter, lag,
                                    recombination_rate, verbose)
    else:
        sample_tab = parallel_gibbs_sampling(loci_tab, n_burn, n_iter, lag,
                                             n_thread, n_gibbs,
                                             recombination_rate, verbose)
    t1 = time.time() - t0
    # posterior
    post_tab = extract_posterior(sample_tab)
    # infer origin
    out1 = infer_origin(seq_data_tab, post_tab, parent = "mother")
    out2 = infer_origin(seq_data_tab, post_tab, parent = "father")
    # merge with input data and intermediate results
    out = pds.merge(
        init_allele_origin_tab[[
            'chrom', 'pos', 'mother_gt', 'father_gt', 'cfdna_gt', 'cfdna_ad',
            'cfdna_dp', 'fetal_fraction',
        ]], out1, how='right', on=['chrom', 'pos']
    )
    out = pds.merge(out, out2,
                    how='left', on=['chrom', 'pos'])
    # flush progress bar
    if verbose:
        time.sleep(2)
        print("\n" * n_gibbs, flush = True)
        sys.stdout.flush()
        sys.stderr.flush()
        print("\n" * n_gibbs, flush = True)
        sys.stdout.flush()
        sys.stderr.flush()
    # short output
    print("Inference done in {} (h:m:s)"
          .format(str(datetime.timedelta(seconds=int(t1)))))
    print_region_origin(out, parent = "mother", threshold = 0.8)
    print_region_origin(out, parent = "father", threshold = 0.8)
    print("")
    # save ?
    if filename is not None:
        try:
            out.to_csv(filename, index=False, sep=";")
        except BaseException:
            raise ValueError("'filename' argument is not a valid file name")
    # output
    return out


# example
if __name__ == '__main__':
    from prediag.fetal_fraction import estimate_global_fetal_fraction
    from prediag.fetal_genotype import infer_global_fetal_genotype
    import prediag.heuristic_fetal_allele_origin as heuristic
    import prediag.simulation as simulation
    from prediag.utils import float2string

    # multi SNP
    seq_length = 0.1
    snp_dist = 1e-3
    phased = True
    ff = 0.2
    ff_constant = False
    recombination_rate = 1.2e-8
    coverage = 200
    coverage_constant = False
    add_noise = True
    verbose = False

    simu_data = simulation.multi_snp_data(seq_length, snp_dist, phased,
                                          ff, ff_constant, recombination_rate,
                                          coverage, coverage_constant,
                                          add_noise, verbose)

    print("fetal fraction estimation")
    fetal_fraction_tab = estimate_global_fetal_fraction(simu_data, tol = 0.05)
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

    print("fetal allele origin inference (init)")
    init_allele_origin_tab = heuristic.infer_parental_allele_origin(
        fetal_genotype_tab.dropna(), recombination_rate = 1.2e-8,
        genetic_dist_threshold = 1e-2, verbose = False, index_version = True
    )

    print("fetal allele origin inference (bayesian)")
    allele_origin_tab = infer_parental_allele_origin(
        simu_data, init_allele_origin_tab, n_burn = 100, n_iter = 400,
        lag = 10, n_thread = 0, n_gibbs = 4, recombination_rate = 1.2e-8,
        both_parent_phased = True, verbose = True
    )

    allele_origin_tab = pds.merge(allele_origin_tab,
                    simu_data[['chrom', 'pos', 'true_allele_origin']],
                    how='left', on=['chrom', 'pos'])
    allele_origin_tab['true_allele_origin'] = \
        allele_origin_tab['true_allele_origin'].apply(readable_allele_origin)
    print(allele_origin_tab.to_string(
        float_format = float2string,
        formatters = {'allele_origin_conf': float2string,
                      'fetal_gt_posterior': float2string}
    ))

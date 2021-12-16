#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
import numpy as np
import pandas as pds
import re
from scipy.spatial.distance import cdist
# internal
from prediag.utils import is_het, is_phased, parse_gt, readable_allele_origin

def infer_single_parental_allele_origin(
    fetal_genotype_tab, target = "mother", recombination_rate = 1.2e-8,
    genetic_dist_threshold = 1e-2, verbose = False
):
    """Infer fetal allele origin among parental (phased) haplotypes based on
    SNP-based heuristic, for a single parent

    Maternal and paternal phased genotype (haplotypes): '0|0', '0|1', '1|0', '1|1'
    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele

    Fetal allele origin in parental haplotypes:
        - 'mat1-pat1', 'mat1-pat2', 'mat2-pat1', 'mat2-pat2' (lexicographic order)
        - with following convention: 'matA-patB' where A = maternal haplotype
        origin, and B = paternal haplotype origin, with 1 = haplotype 1 and
        2 = haplotype 2, i.e. if parental haplotypes = "x|y", 1 corresponds to x,
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

    Procedure:
        1) use predicted fetal genotype to find fetal allele origin at
           parental heterozygous loci (not ambiguous)
        2) use neighboor heterozygous loci to predict the fetal allele origin
           at parental homozygous loci (ambiguous) by a vote in the locus
           neighborhood

    Input:
        fetal_genotype_tab (Pands.DataFrame): table returned by function
            `prediag.fetal_genotype.infer_global_fetal_genotype_vcf`.
        target (string): which parent ('mother', 'father') is targeted.
        recombination_rate (float): average rate of recombination used to
            compute genetic distance between loci. Default value is
            1.2e-8 recomb./bp.
        genetic_dist_threshold (foat): threshold for genetic distance to
            define a locus neighborhood (for ambiguous loci) and infer allele
            origin based on a vote of unambiguous loci around the considered
            locus. Default value is 1cM (centi-Morgan), i.e ~850 kbp with
            default recombination rate.
        verbose (bool): set verbosity.

    Output: Pandas.DataFrame with for each SNP
        * chrom (string): chromosome
        * pos (integer): position on the sequence.
        * <target>_gt (string): `target` genotype 'x/y' with x, y in {0, 1}.
        * <target>_origin (int): index of parental haplotype origin,
            0 for allele 'x' and 1 for allele 'y' in targeted parental
            haplotypes 'x|y'.
        * '<target>_origin_conf' (float vector of length 2): inferred haplotype
            origin confidence, respectively for allele 1 and 2, i.e.
            respectively 'x' and 'y' in targeted parental haplotypes 'x|y'.
    """

    if target not in ['mother', 'father']:
        raise ValueError("target input parameter should be 'mother' or 'father'.")

    # chromosome list (hash table)
    chrom_list = fetal_genotype_tab['chrom'].value_counts(sort=False)

    # output
    origin = []
    ## manage each chromosome independently
    for chrom in chrom_list.keys():
        # chromosome loci
        chrom_loci = (fetal_genotype_tab['chrom'] == chrom).to_numpy()
        # phased loci
        phased_loci = fetal_genotype_tab[target+'_gt'].apply(is_phased).to_numpy()
        # heterozygous loci (unambiguous)
        het_loci = fetal_genotype_tab[target+'_gt'].apply(
            lambda gt : is_het(parse_gt(gt))
        ).to_numpy()
        # if not empty
        if np.sum(het_loci * phased_loci * chrom_loci) > 0:
            ## heterozygous loci haplotype origin
            tmp_origin = []
            # find parental haplotype origin
            for ind in np.nditer(np.where(het_loci * phased_loci * chrom_loci)):
                # context
                chrom = fetal_genotype_tab['chrom'].to_numpy()[ind]
                pos = fetal_genotype_tab['pos'].to_numpy()[ind]
                parent_gt = fetal_genotype_tab[target+'_gt'].to_numpy()[ind]
                fetal_gt = fetal_genotype_tab['fetal_gt_pred'].to_numpy()[ind]
                # find parental haplotype origin
                hap = float(
                    np.where(
                        parse_gt(parent_gt) ==
                        parse_gt(fetal_gt)[0 if target == 'mother' else 1]
                    )[0][0]
                )
                tmp_origin.append(
                    [chrom, pos, parent_gt, fetal_gt, int(hap), [1 - hap, hap]]
                )
            # format output
            het_hap_origin = pds.DataFrame(
                tmp_origin,
                columns=['chrom', 'pos', 'parent_gt', 'fetal_gt',
                         'parent_origin', 'origin_conf']
            )
            ## homozygous loci haplotype origin
            # find parental haplotype origin
            for ind in np.nditer(np.where(phased_loci * chrom_loci)):
                # context
                chrom = fetal_genotype_tab['chrom'].to_numpy()[ind]
                pos = fetal_genotype_tab['pos'].to_numpy()[ind]
                parent_gt = fetal_genotype_tab[target+'_gt'].to_numpy()[ind]
                fetal_gt = fetal_genotype_tab['fetal_gt_pred'].to_numpy()[ind]
                # heterozygous ?
                if het_loci[ind]:
                    origin.append(
                        het_hap_origin.to_numpy()[het_hap_origin['pos'] == pos].tolist()[0]
                    )
                else:
                    # chromosome SNPs
                    candidate_snp = het_hap_origin[np.in1d(
                                            het_hap_origin.chrom,
                                            np.array([chrom]))]['pos']
                    # SNP distances
                    snp_dist = cdist(
                        np.array(pos).reshape(1,1),
                        np.array(candidate_snp).reshape(-1,1)
                    ).reshape(-1)
                    # neighbor SNPs
                    valid_snp = snp_dist * recombination_rate < genetic_dist_threshold
                    # vote
                    vote = het_hap_origin[valid_snp]['parent_origin'].value_counts(sort=False)
                    tmp_allele_origin = np.array(vote.keys())
                    vote = np.array(vote)
                    if len(vote) < 2:
                        vote = np.concatenate([vote, [0.]]) \
                                    if 0 in tmp_allele_origin \
                                    else np.concatenate([[0.], vote])
                    total_vote = np.sum(vote)
                    origin.append([chrom, pos, parent_gt, fetal_gt,
                                   int(vote.argmax()), list(vote/total_vote)])

    df = pds.DataFrame(
        origin,
        columns=['chrom', 'pos', target+'_gt', 'fetal_gt', target+'_origin',
                 target+'_origin_conf']
    )

    return df


def infer_parental_allele_origin(
    fetal_genotype_tab, recombination_rate = 1.2e-8,
    genetic_dist_threshold = 1e-2, verbose = False,
    index_version = False
):
    """Infer fetal allele origin among parental (phased) haplotypes based on
    SNP-based heuristic, for both parents

    Maternal and paternal phased genotype (haplotypes): '0|0', '0|1', '1|0', '1|1'
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

    Procedure:
        1) use predicted fetal genotype to find fetal allele origin at
           parental heterozygous loci (not ambiguous)
        2) use neighboor heterozygous loci to predict the fetal allele origin
           at parental homozygous loci (ambiguous) by a vote in the locus
           neighborhood

    Input:
        fetal_genotype_tab (Pands.DataFrame): table returned by function
            `prediag.fetal_genotype.infer_global_fetal_genotype_vcf`.
        recombination_rate (float): average rate of recombination used to
            compute genetic distance between loci. Default value is
            1.2e-8 recomb./bp.
        genetic_dist_threshold (foat): threshold for genetic distance.
            Default value is 1cM (centi-Morgan), i.e ~850 kbp with default
            recombination rate.
        verbose (bool): set verbosity.
        index_version (bool): if True, return index version of parental
            allele origin (i.e. 'a-b' with a, b in {0, 1}). If False (default),
            return human readable version of parental allele origin
            (i.e. 'matA-patB' with A, B in {1, 2}).

    Output: Pandas.DataFrame with for each SNP
        * chrom (string): chromosome
        * pos (integer): position on the sequence.
        * column of `fetal_genotype_tab` input argument.
        * allele_origin (string): fetal allele origin in parental haplotypes,
            i.e. 'matA-patB' with A, B in {1, 2} if not index version,
            'a-b' with a, b in {0, 1} otherwise.
        * allele_origin_conf (list of float list): vector of confidence
            probabilities for each allele origin '1' vs '2' and each parent
            (first mother, then father).
    """

    # parental orgininal haplotype
    mother_hap_origin = infer_single_parental_allele_origin(
        fetal_genotype_tab.dropna(), "mother", recombination_rate,
        genetic_dist_threshold, verbose
    )
    father_hap_origin = infer_single_parental_allele_origin(
        fetal_genotype_tab.dropna(), "father", recombination_rate,
        genetic_dist_threshold, verbose
    )

    # merge output
    out = pds.merge(
        fetal_genotype_tab,
        mother_hap_origin[['chrom', 'pos', 'mother_origin', 'mother_origin_conf']],
        how='left', on=['chrom', 'pos']
    )
    out = pds.merge(
        out,
        father_hap_origin[['chrom', 'pos', 'father_origin', 'father_origin_conf']],
        how='left', on=['chrom', 'pos']
    )
    # allele origin
    out['allele_origin'] = \
        out['mother_origin'].apply(
            lambda x: str(int(x)) if not np.isnan(x) else ''
        ) \
        + '-' \
        + out['father_origin'].apply(
            lambda x: str(int(x)) if not np.isnan(x) else ''
    )
    if not index_version:
        out['allele_origin'] = out['allele_origin'].apply(readable_allele_origin)

    # allele origin confidence probabilities
    out['allele_origin_conf'] = out[['mother_origin_conf',
                                     'father_origin_conf']].values.tolist()

    out.drop(['mother_origin', 'father_origin',
              'mother_origin_conf', 'father_origin_conf'], inplace=True, axis=1)

    return out


# example
if __name__ == '__main__':
    from prediag.fetal_fraction import estimate_global_fetal_fraction
    from prediag.fetal_genotype import infer_global_fetal_genotype
    import prediag.simulation as simulation
    from prediag.utils import float2string

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

    # parental orgininal haplotype
    print("mother allele inheritance")
    mother_hap_origin = infer_single_parental_allele_origin(
        fetal_genotype_tab.dropna(), target = "mother",
        recombination_rate = 1.2e-8, genetic_dist_threshold = 1e-2,
        verbose = False
    )
    print(mother_hap_origin.to_string(
        float_format = float2string,
        formatters = {'mother_origin_conf': float2string}
    ))

    print("father allele inheritance")
    father_hap_origin = infer_single_parental_allele_origin(
        fetal_genotype_tab.dropna(), target = "father",
        recombination_rate = 1.2e-8, genetic_dist_threshold = 1e-2,
        verbose = False
    )
    print(father_hap_origin.to_string(
        float_format = float2string,
        formatters = {'father_origin_conf': float2string}
    ))

    # merge output
    print("merged output")
    fetal_genotype_tab = infer_parental_allele_origin(
        fetal_genotype_tab.dropna(), recombination_rate = 1.2e-8,
        genetic_dist_threshold = 1e-2, verbose = False
    )
    fetal_genotype_tab = pds.merge(fetal_genotype_tab,
                    simu_data[['chrom', 'pos', 'true_allele_origin']],
                    how='left', on=['chrom', 'pos'])
    fetal_genotype_tab['true_allele_origin'] = \
        fetal_genotype_tab['true_allele_origin'].apply(readable_allele_origin)
    print(fetal_genotype_tab.to_string(
        float_format = float2string,
        formatters = {'allele_origin_conf': float2string,
                      'fetal_gt_posterior': float2string}
    ))

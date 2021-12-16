#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
import numpy as np
import pandas as pd
# internal
import prediag.hap_model as hap_model
from prediag.utils import is_het, is_phased, parse_allele_origin, parse_gt, unparse_allele_origin, unparse_gt


def ar_signal(n_samples, corr=0.9, mu=0, sigma=0.5):
    """Generate auto-regressive AR(1) signal

    Ref: <https://stackoverflow.com/questions/33898665/python-generate-array-of-specific-autocorrelation/33904277#33904277>
    """
    assert 0 < corr < 1, "Auto-correlation must be between 0 and 1"

    # Find out the offset `c` and the std of the white noise `sigma_e`
    # that produce a signal with the desired mean and variance.
    # See https://en.wikipedia.org/wiki/Autoregressive_model
    # under section "Example: An AR(1) process".
    c = mu * (1 - corr)
    sigma_e = np.sqrt((sigma ** 2) * (1 - corr ** 2))

    # Sample the auto-regressive process.
    signal = [c + np.random.normal(0, sigma_e)]
    for _ in range(1, n_samples):
        signal.append(c + corr * signal[-1] + np.random.normal(0, sigma_e))

    return np.array(signal)


def single_snp_data(mother_gt, father_gt, allele_origin, ff = 0.2,
                    coverage = 100, add_noise = True, verbose = False):
    """Simulate single SNP sequencing data

    Simulate: fetal genotype, reads count for mother, father and cfdna.

    Maternal and paternal
        - phased haplotype: '0|0', '0|1', '1|0' '1|1'
        - unphased genotype: '0/0', '0/1', '1/1'

    Fetal allele origin in parental haplotypes:
        - '0-0', '0-1', '1-0', '1-1'
        - with following convention: 'A-B' where A = maternal haplotype origin,
        and B = paternal haplotype origin, 0 = haplotype 1 and 1 = haplotype 2,
        i.e. if parental haplotype = 'x|y', 0 corresponds to x, and 1 to y.

    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele

    Input:
        mother_gt (string): maternal phased haplotype, i.e. 'x|y', or
            maternal unphased genotype, i.e. 'x/y', with x, y in {0, 1}.
        father_gt (string): paternal phased haplotype, i.e. 'x|y', or
            paternal unphased genotype, i.e. 'x/y', with x, y in {0, 1}.
        allele_origin (string): fetal allele origin in parental haplotypes
            'a-b' with a, b in {0, 1}. Only relevant if parental phased
            halotypes are provided.
        ff (float): fetal fraction, between 0 and 1, default value is 20% (0.2).
        coverage (int): sequencing coverage (i.e. average number single read
            copies). Default value is 100.
        add_noise (bool): add noise when partitioning '0' and '1' reads.
            Default value is True.
        verbose (bool): verbosity. Default is False.

    Output:
        fetal_gt (string): fetal genotype 'x/y' with x, y in {0, 1}.
        cfdna_gt (string): cfDNA genotype 'x/y' with x, y in {0, 1}.
        cfdna_ad (np.array): vector of allelic depth (read counts in cfDNA).
    """

    mat_gt = parse_gt(mother_gt)
    pat_gt = parse_gt(father_gt)

    ## allele_origin
    if allele_origin is not None:
        allele_origin = parse_allele_origin(allele_origin)

    ## parental allele
    mat_allele = allele_origin[0] if is_phased(mother_gt) and \
                                     allele_origin is not None and \
                                     allele_origin[0] is not None \
                        else np.random.choice([0, 1])
    pat_allele = allele_origin[1] if is_phased(father_gt) and \
                                     allele_origin is not None and \
                                     allele_origin[1] is not None \
                        else np.random.choice([0, 1])

    ## fetal genotype
    fetal_gt = np.array([mat_gt[mat_allele], pat_gt[pat_allele]])

    ## cfdna genotype
    cfdna_gt = np.sort(np.unique(np.concatenate([mat_gt, fetal_gt])))
    # if mother and child are homozygous
    if len(cfdna_gt) < 2:
        cfdna_gt = np.repeat(cfdna_gt[0], 2)

    ## read count
    n_read = int(np.random.poisson(coverage))

    ## reads from mother and child
    n_mat_read = int(np.round((1-ff) * n_read))
    n_fetal_read = n_read - n_mat_read

    ## distribution of maternal reads
    n_mat_read0 = int(np.round(list(mat_gt).count('0')/2 * n_mat_read))
    n_mat_read1 = n_mat_read - n_mat_read0
    # small noise if heterozygous
    if is_het(mat_gt) and add_noise:
        noise = int(np.random.choice([-1,1])
                    * np.random.randint(np.round(coverage/20), size=1))
        n_mat_read0 += noise
        n_mat_read1 -= noise

    ## distribution of fetal reads
    n_fetal_read0 = int(np.round(list(fetal_gt).count('0')/2 * n_fetal_read))
    n_fetal_read1 = n_fetal_read - n_fetal_read0
    # small noise if heterozygous
    if is_het(fetal_gt) and add_noise:
        noise = int(np.random.choice([-1,1])
                    * np.random.randint(np.round(coverage/20), size=1))
        n_fetal_read0 += noise
        n_fetal_read1 -= noise

    ## allelic depth in cfdna
    cfdna_ad = [n_fetal_read0 + n_mat_read0, n_fetal_read1 + n_mat_read1]

    ## verbosity
    if verbose:
        print("maternal genotype = {} -- paternal genotype = {}"
                .format(mother_gt, father_gt))
        print("fetal genotype = {}".format(unparse_gt(fetal_gt, sort_out = False)))
        print("cfdna genotype = {}".format(unparse_gt(cfdna_gt)))
        print("cfdna allelic depth: {} (0) and {} (1) over {} reads"
                .format(cfdna_ad[0], cfdna_ad[1], np.sum(cfdna_ad)))
        print("Fetal fraction = {} ({} maternal reads -- {} fetal reads)"
                .format(ff, n_mat_read, n_fetal_read))
        print("Details: mother {} (0) and {} (1) -- child {} (0) and {} (1)"
                .format(n_mat_read0, n_mat_read1, n_fetal_read0, n_fetal_read1))

    ## Output
    return unparse_gt(fetal_gt, sort_out = False), unparse_gt(cfdna_gt), cfdna_ad


def multi_snp_data(seq_length = 150, snp_dist=10e-3, phased = False,
                   ff = 0.2, ff_constant = True, recombination_rate = 1.2e-8,
                   coverage = 100, coverage_constant = True,
                   add_noise = True, verbose = False):
    """Simulate SNP sequencing data at chromosome scale

    Simulate: fetal genotype, reads count for mother, father and cfdna.

    Maternal and paternal
        - phased haplotype: '0|0', '0|1', '1|0' '1|1'
        - unphased genotype: '0/0', '0/1', '1/1'

    Fetal genotype:
        - '0/0', '0/1', '1/0', '1/1' (lexicographic order)
        - with following convention: 'A/B' where A = maternal allele and
        B = paternal allele

    Input:
        seq_length (float): length of simulated sequence (in Mbp).
        snp_dist (float): average inter-SNP distance (in Mbp).
        phased (bool): generate parental phased haplotype (if True) or
            unphased genotype (if False). Default value is False.
        ff (float): fetal fraction. If list, variable
            fetal fraction.
        ff_constant (bool): should the fetal fraction be constant. If False,
            fetal fraction is generated with an auto-regressive signal of
            overage ff. Default is True.
        recombination_rate (float): recombination rate per bp (between 0 and 1).
            Default value 1.2e-8.
        coverage (int): sequencing coverage (i.e. average number of
            single read copies). If list, variable coverage.
        coverage_constant (bool): should the coverage be constant. If False,
            coverage is generated with an auto-regressive signal of
            overage ff. Default is True.
        add_noise (bool): add noise when partitioning '0' and '1' reads.
            Default value is True.
        verbose (bool): verbosity. Default is False.

    Output: Pandas.DataFrame with for each SNP
        * chrom (string): chromosome
        * pos (integer): position on the sequence.
        * mother_gt (string): maternal haplotype 'x|y' with x, y in {0,1},
            or maternal genotype 'x/y' if haplotype not available.
        * father_gt (string): paternal haplotype 'x|y' with x, y in {0,1},
            or paternal genotype 'x/y' if haplotype not available.
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
        * true_ff (float): fetal fraction.
        * coverage (int): theoretical coverage (i.e. average read counts).
        * fetal_gt (string): fetal genotype 'x/y' with x, y in {0, 1}.
        * cfdna_gt (string): cfDNA genotype 'x/y' with x, y in {0, 1}.
        * cfdna_ad (int list): cfDNA allelic depth (= read count) per allele.
        * cfdna_dp (int): cfDNA coverage (= total read count) on the locus.
        * true_allele_origin (string): fetal allele origin in parental haplotypes
            'a-b' with a, b in {0, 1}. Only relevant if parental phased
            halotypes are provided.
    """

    ## generate SNP position
    n_snp = int(seq_length / snp_dist)
    uniform_pos = np.random.uniform(size=int(n_snp))
    snp_pos = np.sort(np.round(uniform_pos * seq_length * 1e6).astype(int))

    ## fetal fraction
    ff_values = None
    if ff_constant:
        ff_values = np.repeat(ff, n_snp)
    else:
        tmp = ar_signal(n_snp, corr=0.9, mu=0, sigma=0.5)
        ff_values = ff + 0.5 * ff * tmp/np.max(np.abs(tmp))

    ## coverage
    coverage_values = None
    if coverage_constant:
        coverage_values = np.repeat(coverage, n_snp)
    else:
        tmp = ar_signal(n_snp, corr=0.9, mu=0, sigma=0.5)
        coverage_values = np.round(coverage
                + 0.8 * coverage * tmp/np.max(np.abs(tmp))).astype(int)

    ## possible alleles
    possible_alleles = ['0', '1']

    ## phasing error probability is low
    mother_phasing_error_proba = 1e-10
    father_phasing_error_proba = 1e-10

    ## possible allele origin
    possible_allele_origin = ['0-0', '0-1', '1-0', '1-1']

    ## SNP generation
    out = []
    previous_pos = 0
    previous_allele_origin = None
    for i, (pos, ff, cov) in enumerate(zip(snp_pos, ff_values, coverage_values)):
        ## moter genotype
        mother_gt = unparse_gt(np.random.choice(possible_alleles, size=2),
                               phased = phased)
        ## father genotype
        father_gt = unparse_gt(np.random.choice(possible_alleles, size=2),
                               phased = phased)
        ## recombination
        locus_dist = np.abs(pos - previous_pos)
        allele_origin_proba = hap_model.haplotype_transition_probability(
                                    previous_allele_origin,
                                    mother_phasing_error_proba,
                                    father_phasing_error_proba,
                                    locus_dist, recombination_rate,
                                    mother_phased = is_phased(mother_gt),
                                    father_phased = is_phased(father_gt))

        allele_origin = np.random.choice(possible_allele_origin,
                                         p=allele_origin_proba)
        # next locus
        previous_allele_origin = allele_origin
        previous_pos = pos

        ## simulation
        fetal_gt, cfdna_gt, cfdna_ad = \
                single_snp_data(mother_gt, father_gt, allele_origin, ff, cov,
                                verbose)
        ## phasing Value
        mother_pq = 1e-12
        mother_jq = 1e-12
        father_pq = 1e-12
        father_jq = 1e-12
        ##
        out.append(['chr00', pos, mother_gt, father_gt, mother_pq, mother_jq,
                    father_pq, father_jq, ff, cov, fetal_gt, cfdna_gt,
                    cfdna_ad, np.sum(cfdna_ad), allele_origin])

    df = pd.DataFrame(out, columns=['chrom', 'pos', 'mother_gt',
                                    'father_gt', 'mother_pq', 'mother_jq',
                                    'father_pq', 'father_jq', 'true_ff',
                                    'coverage', 'true_fetal_gt', 'cfdna_gt',
                                    'cfdna_ad', 'cfdna_dp', 'true_allele_origin'])
    return df


# example
if __name__ == '__main__':

    from prediag.utils import float2string

    # single SNP
    mother_gt = '0/1'
    father_gt = '1/1'
    allele_origin = None
    ff = 0.2
    coverage = 100
    add_noise = True
    verbose = True

    fetal_gt, cfdna_gt, cfdna_ad = single_snp_data(mother_gt, father_gt,
                                                   allele_origin, ff, coverage,
                                                   add_noise, verbose)

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

    chrom = multi_snp_data(seq_length, snp_dist, phased,
                           ff, ff_constant, recombination_rate,
                           coverage, coverage_constant,
                           add_noise, verbose)

    print(chrom.to_string(float_format = float2string))

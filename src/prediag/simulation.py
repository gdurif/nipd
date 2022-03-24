#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
import numpy as np
import pandas as pd
# internal
import prediag.hap_model as hap_model
from prediag.utils import is_het, is_phased, parse_allele_origin, parse_gt, unparse_allele_origin, unparse_gt, readable_allele_origin


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
    
    # RNG
    rng = np.random.default_rng()

    # Sample the auto-regressive process.
    signal = [c + rng.normal(0, sigma_e)]
    for _ in range(1, n_samples):
        signal.append(c + corr * signal[-1] + rng.normal(0, sigma_e))

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
        add_noise (bool): add noise into allelic depth (of average level 1/20 
            of the coverage). Default value is True.
        verbose (bool): verbosity. Default is False.

    Output:
        fetal_gt (string): fetal genotype 'x/y' with x, y in {0, 1}.
        cfdna_gt (string): cfDNA genotype 'x/y' with x, y in {0, 1}.
        cfdna_ad (np.array): vector of allelic depth (read counts in cfDNA).
    """

    mat_gt = parse_gt(mother_gt)
    pat_gt = parse_gt(father_gt)
    
    # RNG
    rng = np.random.default_rng()

    ## allele_origin
    if allele_origin is not None:
        allele_origin = parse_allele_origin(allele_origin)

    ## parental allele
    mat_allele = allele_origin[0] \
        if is_phased(mother_gt) and \
            allele_origin is not None and \
            allele_origin[0] is not None \
        else rng.choice([0, 1])
    pat_allele = allele_origin[1] \
        if is_phased(father_gt) and \
            allele_origin is not None and \
            allele_origin[1] is not None \
        else rng.choice([0, 1])

    ## fetal genotype
    fetal_gt = np.array([mat_gt[mat_allele], pat_gt[pat_allele]])

    ## cfdna genotype
    cfdna_gt = np.sort(np.unique(np.concatenate([mat_gt, fetal_gt])))
    # if mother and child are homozygous
    if len(cfdna_gt) < 2:
        cfdna_gt = np.repeat(cfdna_gt[0], 2)

    ## read count
    n_read = int(rng.poisson(coverage))

    ## reads from mother and child
    n_mat_read = int(np.round((1-ff) * n_read))
    n_fetal_read = n_read - n_mat_read

    ## distribution of maternal reads
    n_mat_read0 = int(np.round(list(mat_gt).count('0')/2 * n_mat_read))
    n_mat_read1 = n_mat_read - n_mat_read0
    # small noise if heterozygous
    if is_het(mat_gt) and add_noise:
        noise = int(rng.normal(0, np.round(coverage/20)))
        n_mat_read0 += noise
        n_mat_read1 -= noise

    ## distribution of fetal reads
    n_fetal_read0 = int(np.round(list(fetal_gt).count('0')/2 * n_fetal_read))
    n_fetal_read1 = n_fetal_read - n_fetal_read0
    # small noise if heterozygous
    if is_het(fetal_gt) and add_noise:
        noise = int(rng.normal(0, np.round(coverage/20)))
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


def simulate_allele_origin(seq_length, recombination_rate, snp_pos):
    """Simulate fetal allele origin for a given parent
    
    Convention: 0 = haplotype 1 and 1 = haplotype 2

    Input:
        seq_length (float): length of simulated sequence (in bp).
        recombination_rate (float): recombination rate in cM/bp.  If None, 
        no recombination is simulated.
        snp_pos (array of float): arrays of SNP positions (in bp).

    Output: {0,1} valued vector of allele origin of length corresponging to
        `snp_pos` length.
    """
    
    # RNG
    rng = np.random.default_rng()
    
    # allele origin for the first locus
    first_allele_origin = rng.choice([0, 1])
    
    # vector of allele origin (without recombination)
    allele_origin = np.repeat(first_allele_origin, len(snp_pos))
    
    # any recombination ?
    if recombination_rate is not None:
        ## generate recombination event if any
        recomb_event = rng.binomial(
            n = 1, p = min(seq_length * recombination_rate * 0.01, 1)
        )
        # in case of a recombination event ?
        if recomb_event:
            # recombination position
            recomb_pos = rng.uniform(snp_pos.min(), snp_pos.max())
            # vector of allele origin (with recombination)
            allele_origin = np.concatenate([
                np.repeat(first_allele_origin, np.sum(snp_pos <= recomb_pos)),
                np.repeat(1 - first_allele_origin, np.sum(snp_pos > recomb_pos))
            ])
    
    # output
    return allele_origin
    

def multi_snp_data(seq_length = 150, snp_dist=2e-3, phased = False,
                   ff = 0.2, ff_constant = True, recombination_rate = 1.2,
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
    
    Fetal allele origin in parental haplotypes:
        - '0-0', '0-1', '1-0', '1-1'
        - with following convention: 'A-B' where A = maternal haplotype origin,
        and B = paternal haplotype origin, 0 = "haplotype 1" 
        and 1 = "haplotype 2", i.e. if parental haplotype = 'x|y', 
        then 0 corresponds to allele x, and 1 to allele y.
    
    Between-SNP distances are simulated through an exponential distribution of 
    parameter `1/snp_dist`.

    Input:
        seq_length (float): length of simulated sequence (in Mbp).
        snp_dist (float): average inter-SNP distance (in Mbp). Default is 
            `0.002 Mb`, i.e. `2 kb`.
        phased (bool): generate parental phased haplotype (if True) or
            unphased genotype (if False). Default value is False.
        ff (float): fetal fraction. If list, variable
            fetal fraction.
        ff_constant (bool): should the fetal fraction be constant. If False,
            fetal fraction is generated with an auto-regressive signal of
            overage ff. Default is True.
        recombination_rate (float): recombination rate in cM/Mbp 
            Default value 1.2. If None, no recombination is simulated.
        coverage (int): sequencing coverage (i.e. average number of
            single read copies). If list, variable coverage.
        coverage_constant (bool): should the coverage be constant. If False,
            coverage is generated with an auto-regressive signal of
            overage ff. Default is True.
        add_noise (bool): add noise into allelic depth (of average level 1/20 
            of the coverage). Default value is True.
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
            (c.f. 10x-genomics doc). 0 by default (no phasing error).
        * mother_jq (float): mother junction quality probability, "probability
            that there is a large-scale phasing switch error occuring between
            this variant and the following variant" (10x-genomics doc).
            0 by default (no phasing error).
        * father_pq (float): father phasing quality probability, "probability
            that alleles are phased incorrectly in a heterozygous call"
            (10x-genomics doc). 0 by default (no phasing error).
        * father_jq (float): father junction quality probability, "probability
            that there is a large-scale phasing switch error occuring between
            this variant and the following variant" (10x-genomics doc).
            0 by default (no phasing error).
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
    
    # RNG
    rng = np.random.default_rng()

    ## generate SNP position (in bp)
    snp_dist_vec = rng.exponential(
        scale = snp_dist, 
        size = 5*int(seq_length / snp_dist)
    )
    
    snp_pos = np.cumsum(snp_dist_vec)
    snp_pos = (1E6*np.around(snp_pos[snp_pos <= seq_length], decimals=6)).astype(int)
    n_snp = len(snp_pos)

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
        coverage_values = np.round(
            coverage + 0.8 * coverage * tmp / np.max(np.abs(tmp))
        ).astype(int)

    ## possible alleles
    possible_alleles = ['0', '1']

    ## phasing error probability is low (not used at the moment)
    mother_phasing_error_proba = 1e-10
    father_phasing_error_proba = 1e-10

    ## possible allele origin
    possible_allele_origin = ['0-0', '0-1', '1-0', '1-1']
    
    ## simulate allele origins along the region for both parents
    # (with potential recombination event)
    mother_allele_origin = simulate_allele_origin(
        int(seq_length*1E6), recombination_rate*1E-6, snp_pos
    )
    father_allele_origin = simulate_allele_origin(
        seq_length, recombination_rate, snp_pos
    )

    ## SNP generation
    out = []
    for i, (pos, ff, cov, mat_ori, pat_ori) in \
        enumerate(zip(
            snp_pos, ff_values, coverage_values, 
            mother_allele_origin, father_allele_origin
        )):
        ## moter genotype
        mother_gt = unparse_gt(
            rng.choice(possible_alleles, size=2), phased = phased
        )
        ## father genotype
        father_gt = unparse_gt(
            rng.choice(possible_alleles, size=2), phased = phased
        )
        ## allele origin
        allele_origin = f"{mat_ori}-{pat_ori}"

        ## simulation
        fetal_gt, cfdna_gt, cfdna_ad = single_snp_data(
            mother_gt, father_gt, allele_origin, ff, cov, verbose
        )
        
        ## phasing Value (0 by default, no phasing error in simulation)
        mother_pq = 0
        father_pq = 0
        mother_jq = 0
        father_jq = 0
        ##
        out.append(['chr00', pos, mother_gt, father_gt, mother_pq, mother_jq,
                    father_pq, father_jq, ff, cov, fetal_gt, cfdna_gt,
                    cfdna_ad, np.sum(cfdna_ad), allele_origin])
    # format data
    df = pd.DataFrame(out, columns=['chrom', 'pos', 'mother_gt',
                                    'father_gt', 'mother_pq', 'mother_jq',
                                    'father_pq', 'father_jq', 'true_ff',
                                    'coverage', 'true_fetal_gt', 'cfdna_gt',
                                    'cfdna_ad', 'cfdna_dp', 'true_allele_origin'])
    # readable allele origin
    df['true_allele_origin'] = df['true_allele_origin'].apply(readable_allele_origin)
    # output
    return df


# example
if __name__ == '__main__':

    from prediag.utils import float2string
    
    # RNG
    rng = np.random.default_rng()

    # single SNP
    mother_gt = '0/1'
    father_gt = '1/1'
    allele_origin = None
    ff = 0.2
    coverage = 100
    add_noise = True
    verbose = True

    fetal_gt, cfdna_gt, cfdna_ad = single_snp_data(
        mother_gt, father_gt, allele_origin, ff, coverage,
        add_noise, verbose
    )
    
    # allele origin
    seq_length = 10
    recombination_rate = 1.2
    snp_pos = np.sort(rng.uniform(0, seq_length, size = int(seq_length/2e-3)))
    allele_origin = simulate_allele_origin(
        seq_length, recombination_rate, snp_pos
    )

    # multi SNP
    seq_length = 10
    snp_dist = 1.8e-3
    phased = True
    ff = 0.2
    ff_constant = False
    recombination_rate = 1.2
    coverage = 100
    coverage_constant = False
    add_noise = True
    verbose = False

    chrom = multi_snp_data(seq_length, snp_dist, phased,
                           ff, ff_constant, recombination_rate,
                           coverage, coverage_constant,
                           add_noise, verbose)

    print(chrom.to_string(float_format = float2string))

#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
from collections import Iterable
import math
import numpy as np
import pandas as pds
import re
import sys
from textwrap import dedent
import vcf
import vcf.utils
from warnings import warn as warning
# internal
from prediag.utils import is_het, is_phased, parse_gt, unparse_gt


def get_vcf_call(record, samples):
    """Find data (Call) from VCF Record

    Input:
        record (vcf.model._Record): VCF data entry.
        samples (vcf.model._Record.samples): corresponding samples.

    Output: corresponding vcf.model._Call if existing, None otherwise.
    """
    if record is not None:
        sample = find_genotype(record, samples)
        if sample is not None:
            return record.genotype(sample)
    return None


def find_genotype(record, samples):
    """Find genotype in VCF Record

    Input:
        record (vcf.model._Record): VCF data entry.
        samples (vcf.model._Record.samples): corresponding samples.

    Output: genotype "x/y" if existing, None otherwise.
    """
    for sample in samples:
        if not record.genotype(sample)['GT'] == "./.":
            return sample
    return None


def parse_region(regionstr):
    """parse chromosome region

    Input:
        regionstr (string): chromosome region.

    Output: chrom, start, end
    """
    if not re.match("chr([0-9]+|X|Y)([:-][0-9]+)*", regionstr):
        raise ValueError("Chromosome region " + regionstr + " is wrongly formatted")
    region_split = re.split(':|-', regionstr)
    chrom = region_split[0]
    start = int(region_split[1]) if len(region_split) > 1 else None
    end = int(region_split[2]) if len(region_split) > 2 else None
    return chrom, start, end


def phred_quality_score2prob(score):
    """Convert Phred quality score to corresponding probability

    Phred quality score: Q
    Related probability: P

    $Q = -10 \log_{10} P$ and $P = 10^{-Q/10}$

    Source: https://en.wikipedia.org/wiki/Phred_quality_score

    Input (float): phred quality score
    Output (float): related probability
    """
    return 10**(-score/10)

def prob2phred_quality_score(prob):
    """Convert Phred quality score to corresponding probability

    Phred quality score: Q
    Related probability: P

    $Q = -10 \log_{10} P$ and $P = 10^{-Q/10}$

    Source: https://en.wikipedia.org/wiki/Phred_quality_score

    Input (float): phred quality score
    Output (float): related probability
    """
    return -10 * math.log10(prob)


def vcf_coreader(mother_vcf, father_vcf, cfdna_vcf, region=None, **kwargs):
    """Open and jointly walk through multiple VCf files
    """
    # create vcf files iterators
    try:
        mother_reader = vcf.Reader(filename=mother_vcf, encoding='utf8')
        father_reader = vcf.Reader(filename=father_vcf, encoding='utf8')
        cfdna_reader = vcf.Reader(filename=cfdna_vcf, encoding='utf8')
    except BaseException:
        raise ValueError(dedent('''
            warning! could not create iterator for the vcf. The input file
            does probably not contain any variants in the region.'''))

    # samples
    mother_samples = mother_reader.samples
    father_samples = father_reader.samples
    cfdna_samples = cfdna_reader.samples

    # region
    if region is not None:
        chrom, start, end = parse_region(region)
        try:
            mother_reader = mother_reader.fetch(chrom, start, end)
            father_reader = father_reader.fetch(chrom, start, end)
            cfdna_reader = cfdna_reader.fetch(chrom, start, end)
        except ValueError as e:
            errmessage = e.args[0]
            if 'could not create iterator for region' in errmessage:
                sys.exit('warning! ' + errmessage
                         + ', probably the input file does not contain any variants in the region.')

    # coreader
    co_reader = vcf.utils.walk_together(mother_reader, father_reader, cfdna_reader)

    return co_reader, mother_samples, father_samples, cfdna_samples


def allelic_depth(gt, data, correct_genotype = True,
                  min_rel_depth = 0.01, min_abs_depth = 2):
    """Extract allelic depth from VCF data and correct genotype accordingly if
    requested

    Input:
        gt (string): haplotype 'x|y' or genotype 'x/y', with x, y in {0,1}.
        data (vcf.model._Call): VCF data for the corresponding locus.
        correct_genotype (bool): if True genotype is corrected according to
            allelic depths, depending on `min_rel_depth` and `min_abs_depth`.
        min_rel_depth (float): minimum relative threshold between 0 and 1 for
            the ratio 'allelic_depth/coverage' under which the corresponding
            allele is considered not expressed.
        min_rel_depth (integer): minimum absolute threshold for the count
            'allelic_depth' under which the corresponding allele
            is considered not expressed.

    Output:
        gt (string): haplotype 'x|y' or corrected (if requested)
            genotype 'x/y', with x, y in {0,1}.
        ad (int list): cfDNA allelic depth (= read count) per allele at the
            locus (of length 2). In (unphased) genotype, index 0
            corresponds to read count matching reference allele (0)
            and index 1 to alternative allele (1).
        dp (int): cfDNA coverage (= total read count) on the locus.
    """
    # allelic depth
    ad = None
    try:
        ad = [item for item in data['AD'] if item]
        if len(ad) == 0:
            ad = None
    except BaseException:
        pass
    # coverage (combined depth)
    dp = None
    try:
        dp = data['DP']
    except BaseException:
        dp = np.sum(ad)

    # check coverage
    if dp is not None:

        # potential issue with cfDNA homozygous locus
        # missing or size-1 vector or scalar allelic depth
        if ad is None or isinstance(ad, int) or len(ad) == 1:
            # homozygous locus
            if not is_het(parse_gt(gt)):
                # missing allelic depth
                if dp is not None and ad is None:
                    # reformat allelic depth
                    if '0' in parse_gt(gt):
                        ad = [dp, 0]
                    else:
                        ad = [0, dp]
                # size-1 vector or scalar allelic depth
                elif ad is not None and (isinstance(ad, int) or len(ad) == 1):
                    # if scalar
                    if not isinstance(ad, Iterable):
                        ad = [ad]
                    # reformat allelic depth
                    if '0' in parse_gt(gt):
                        ad = ad + [0]
                    else:
                        ad = [0] + ad

            # heterozygous locus = problem!!!!
            else:
                warning(
                    "Problem: heterozygous locus with a length-1 allelic "
                    + "depth record"
                )
                gt, ad, dp = None, None, None

        # filter out poly(>2)-allelic loci
        elif len(ad) > 2 and \
            np.any(np.array(ad[2:], dtype=np.float) > min_rel_depth * dp) and \
            np.any(np.array(ad[2:], dtype=np.float) > min_abs_depth):
            warning(
                "Poly-allelic SNPs (i.e. with more than 2 alleles: "
                + "0,1,2,...) are discarded for the moment."
            )
            gt, ad, dp = None, None, None

    # genotype correction (only for genotypes)
    if correct_genotype and not None in [gt, ad, dp] and not is_phased(gt):
        # heterozygous
        if np.all(np.array(ad, dtype=np.float) > min_rel_depth * dp) and \
            np.all(np.array(ad, dtype=np.float) > min_abs_depth):
            gt = '0/1'
        # homozygous
        elif ad[0] > min_rel_depth * dp and ad[0] > min_abs_depth:
                gt = '0/0'
        elif ad[1] > min_rel_depth * dp and ad[1] > min_abs_depth:
            gt = '1/1'
        # problem
        else:
            warning(
                "Problem: no expressed allele at the locus."
            )
            gt, ad, dp = None, None, None

    # output
    return gt, ad, dp


def phasing_quality(gt, data):
    """Extract phasing quality from VCF data

    Input:
        gt (string): haplotype 'x|y' or genotype 'x/y' if haplotype not
            available,  with x, y in {0,1}.
        data (vcf.model._Call): VCF data for the corresponding locus.

    Output:
        pq (float): phasing quality probability, "probability
            that alleles are phased incorrectly in a heterozygous call"
            (10x-genomics doc).
        jq (float): junction quality probability, "probability
            that there is a large-scale phasing switch error occuring between
            this variant and the following variant" (10x-genomics doc).
    """
    pq = None
    jq = None

    if is_phased(gt) and not is_het(parse_gt(gt)):
        pq = 0
        jq = 0
    else:
        try:
            pq = phred_quality_score2prob(data['PQ'])
        except BaseException:
            pass
        try:
            jq = phred_quality_score2prob(data['JQ'])
        except BaseException:
            pass
    # output
    return pq, jq


def phasing_homozygous(gt):
    """Replace genotype "x/x" by haplotypes "x|x"

    Input:
        gt (string): haplotype 'x|y' or genotype 'x/y' if haplotype not
            available,  with x, y in {0,1}.

    Output: haplotype 'x|y' or genotype 'x/y' if haplotype not
        available and x != y,  with x, y in {0,1}.
    """
    if not is_phased(gt) and not is_het(parse_gt(gt)):
        return unparse_gt(parse_gt(gt), phased = True)
    else:
        return gt



def load_vcf_data(mother_vcf, father_vcf, cfdna_vcf, region = None,
                  filename = None, snp_list_file = None,
                  correct_genotype = True,
                  min_rel_depth = 0.01, min_abs_depth = 2,
                  verbose = False, **kwargs):
    """Load VCF data (mother, father and cfDNA sequencing data) into Pandas
    DataFrame

    Phasing related probabilities: see https://support.10xgenomics.com/genome-exome/software/pipelines/latest/output/vcf

    Input:
        mother_vcf (string): mother VCF file.
        father_vcf (string): father VCF file.
        cfdna_vcf (string): cfdna VCf file.
        region (string): chromosome region "chrA-B-C" where integers A, B and C
            identify the region.
        filename (string): file to save SNP table in csv format.
        snp_list_file (string): name of file with input list of SNPs to
            consider, csv two-column (chromosome and position) format with
            header. If None (default), all SNPs in the region are considered.
        correct_genotype (bool): if True genotype are corrected according to
            allelic depths, depending on `min_rel_depth` and `min_abs_depth`.
        min_rel_depth (float): minimum relative threshold between 0 and 1 for
            the ratio 'allelic_depth/coverage' under which the corresponding
            allele is considered not expressed.
        min_rel_depth (integer): minimum absolute threshold for the count
            'allelic_depth' under which the corresponding allele
            is considered not expressed.
        verbose (bool): verbosity. Default is False.

    Output: Pandas.DataFrame with for each SNP
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
    """
    # vcf file iterators and related samples
    co_reader, mother_samples, \
    father_samples, cfdna_samples = vcf_coreader(mother_vcf, father_vcf,
                                                 cfdna_vcf, region)

    # iterate through vcf files and estimate fetal fraction
    out = []
    for tup in co_reader:
        # record
        mother, father, cfdna = tup
        # locus data for informative sample
        mother_data = get_vcf_call(mother, mother_samples)
        father_data = get_vcf_call(father, father_samples)
        cfdna_data = get_vcf_call(cfdna, cfdna_samples)
        # if data from mother, father and cfdna are available
        if not any(elem is None for elem in [mother_data, father_data, cfdna_data]):

            # # logging
            # print("MOTHER chrom {} pos {}".format(mother.CHROM, mother.POS))
            # print("FATHER chrom {} pos {}".format(father.CHROM, father.POS))
            # print("CFDNA chrom {} pos {}".format(cfdna.CHROM, cfdna.POS))
            # print("-----------")
            # print("MOTHER data")
            # print(mother_data)
            # print("-----------")
            # print("FATHER data")
            # print(father_data)
            # print("-----------")
            # print("CFDNA data")
            # print(cfdna_data)
            # print("-----------")

            # chromosome and position
            chrom = cfdna.CHROM
            pos = cfdna.POS
            # genotype
            mother_gt = mother_data['GT']
            father_gt = father_data['GT']
            cfdna_gt = cfdna_data['GT']

            # allelic depth in plasma
            cfdna_gt, cfdna_ad, cfdna_dp = allelic_depth(
                cfdna_gt, cfdna_data, correct_genotype,
                min_rel_depth, min_abs_depth
            )

            # correct parent genotype if phasing_homozygous
            mother_gt = phasing_homozygous(mother_gt)
            father_gt = phasing_homozygous(father_gt)

            # phasing quality in mother
            mother_pq, mother_jq = phasing_quality(mother_gt, mother_data)
            # phasing quality in father
            father_pq, father_jq = phasing_quality(father_gt, father_data)

            # # logging
            # print("output")
            # tmp = [[chrom, pos, mother_gt, father_gt, cfdna_gt,
            #             cfdna_ad, cfdna_dp, mother_pq, mother_jq,
            #             father_pq, father_jq]]
            #
            # print(pds.DataFrame(tmp,
            #                     columns=[
            #                         'chrom', 'pos',
            #                         'mother_gt', 'father_gt',
            #                         'cfdna_gt', 'cfdna_ad',
            #                         'cfdna_dp',
            #                         'mother_pq', 'mother_jq',
            #                         'father_pq', 'father_jq']).to_string())
            # print("#######################################################")

            # output
            out.append([chrom, pos, mother_gt, father_gt, cfdna_gt,
                        cfdna_ad, cfdna_dp, mother_pq, mother_jq,
                        father_pq, father_jq])

    # format data as dataframe
    loci_tab = pds.DataFrame(out, columns=['chrom', 'pos', 'mother_gt', 'father_gt',
                                           'cfdna_gt', 'cfdna_ad', 'cfdna_dp',
                                           'mother_pq', 'mother_jq',
                                           'father_pq', 'father_jq'])
    # filter by SNP list if provided
    if snp_list_file is not None:
        try:
            snp_list = pds.read_csv(snp_list_file)
            snp_list.columns = ['chrom', 'pos']
            snp_list['pos'] = snp_list['pos'].apply(lambda x:
                                                    x.replace(u'\xa0', u' '))
            snp_list['pos'] = snp_list['pos'].apply(lambda x:
                                                    int(x.replace(' ', '')))
            loci_tab = pds.merge(loci_tab, snp_list,
                                 how='inner', on=['chrom', 'pos'])
        except BaseException:
            raise ValueError(
                "'snp_file_list' argument is not a valid file name "
                + "or not a valid csv file with columns 'chrom' and 'pos'."
            )

    # save if required
    if filename is not None:
        try:
            loci_tab.to_csv(filename, index=False, sep=";")
        except BaseException:
            raise ValueError("'filename' argument is not a valid file name")
    # output
    return loci_tab



# check
if __name__ == '__main__':

    # phred proba and score
    for score in np.log([0.1, 1, 10, 100, 1000]):
        assert phred_quality_score2prob(score) == 10**(-score/10)
    for prob in [1e-15, 1e-10, 1e-5, 1e-1, 0.5, 1-1e-1, 1-1e-5, 1-1e-10, 1-1e-15]:
        prob2phred_quality_score(prob) == -10 * math.log10(prob)

    # allelic depth (easy case)
    gt = '0/1'
    ad = [10,20]
    dp = np.sum(ad)
    data = {'AD': ad, 'DP': dp}

    correct_genotype = True
    min_rel_depth = 0.01
    min_abs_depth = 2

    gt1, ad1, dp1 = allelic_depth(
        gt, data, correct_genotype, min_rel_depth, min_abs_depth
    )

    assert gt1 == gt and ad1 == ad and dp1 == dp

    # allelic depth (homozygous with problem)
    gt = '1/1'
    ad = [50]
    dp = np.sum(ad)
    data = {'AD': ad, 'DP': dp}

    correct_genotype = True
    min_rel_depth = 0.01
    min_abs_depth = 2

    gt1, ad1, dp1 = allelic_depth(
        gt, data, correct_genotype, min_rel_depth, min_abs_depth
    )

    assert gt1 == gt and ad1 == [0] + ad and dp1 == dp

    # allelic depth (heterozygous with problem)
    gt = '0/1'
    ad = [50]
    dp = np.sum(ad)
    data = {'AD': ad, 'DP': dp}

    correct_genotype = True
    min_rel_depth = 0.01
    min_abs_depth = 2

    gt1, ad1, dp1 = allelic_depth(
        gt, data, correct_genotype, min_rel_depth, min_abs_depth
    )

    assert gt1 == None and ad1 == None and dp1 == None

    # allelic depth (need correction)
    gt = '1/1'
    ad = [10, 50]
    dp = np.sum(ad)
    data = {'AD': ad, 'DP': dp}

    correct_genotype = True
    min_rel_depth = 0.01
    min_abs_depth = 2

    gt1, ad1, dp1 = allelic_depth(
        gt, data, correct_genotype, min_rel_depth, min_abs_depth
    )

    assert gt1 == '0/1' and ad1 == ad and dp1 == dp

    # phasing homozygous (x/x becomes x|x)
    assert phasing_homozygous('0/0') == '0|0'
    assert phasing_homozygous('1/1') == '1|1'
    assert phasing_homozygous('0/1') == '0/1'
    assert phasing_homozygous('0|0') == '0|0'
    assert phasing_homozygous('1|1') == '1|1'
    assert phasing_homozygous('0|1') == '0|1'
    assert phasing_homozygous('1|0') == '1|0'

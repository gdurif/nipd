#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

# external
import textwrap
# internal
# from prediag.filter import loci_tab_filter
from prediag.vcf_reader import load_vcf_data


def custom_arg_parser(parser):
    """Common argument parser for command line tools

    Input:
        parser (argparse.ArgumentParser): empty argument parser.

    Output:
        parser (argparse.ArgumentParser): filled argument parser.
    """
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument(
        "-cfdna_vcf", "--cfdna_vcf",
        help = 'Maternal plasma cfDNA genome VCF file'
    )
    requiredNamed.add_argument(
        "-mat_vcf", "--maternal_vcf",
        help = 'Maternal genome VCF file'
    )
    requiredNamed.add_argument(
        "-pat_vcf", "--paternal_vcf",
        help = 'Paternal genome VCF file')
    parser.add_argument(
        "-snp", "--snp_list",
        help=textwrap.dedent('''\
            name of file with input list of SNPs to consider, csv two-column
            (chromosome and position) format with header. If not supplied
            (default), all SNPs in the region are considered.'''),
        default=None
    )
    parser.add_argument(
        "-o", "--output",
        help=textwrap.dedent('''\
            output file (if not supplied, output is printed to stdout only)'''),
        default=None
    )
    parser.add_argument(
        "-r", "--region",
        help=textwrap.dedent('''\
            chromosome region "chrA-B-C" where integers A, B and C
            identify the region. !!! require VCF index files (tbi) !!!'''),
        default=None
    )
    parser.add_argument(
        "-mc", "--min_coverage",
        help=textwrap.dedent('''\
            Minimum coverage (=read count) for a locus to be considered'''),
        type=int, default=50
    )
    parser.add_argument(
        "-mrd", "--min_rel_depth",
        help=textwrap.dedent('''\
            Minimum relative threshold between 0 and 1 for
            the ratio 'allelic_depth/coverage' (at each locus) under which
            the corresponding allele is considered not expressed.'''),
        type=float, default=0.01
    )
    parser.add_argument(
        "-mad", "--min_abs_depth",
        help=textwrap.dedent('''\
            Minimum absolute threshold for the 'allelic_depth' (at each locus)
            under which the corresponding allele is considered not
            expressed.'''),
        type=int, default=2
    )
    parser.add_argument("-v", "--verbose", help="Set verbosity on.",
                        action="store_true")

    return parser


def custom_ff_arg_parser(parser):
    """Argument parser for command line tools depending on fetal
    fraction estimation

    Input:
        parser (argparse.ArgumentParser): argument parser.

    Output:
        parser (argparse.ArgumentParser): filled argument parser.
    """
    parser.add_argument(
        "-ff", "--fetal_fraction_file",
        help=textwrap.dedent('''\
            name of CSV file containing result of fetal fraction estimation,
            as produced by `prediag_fetal_fraction` CLI tool.
            Default is None and fetal fraction estimation tool is run.'''),
        default=None
    )
    parser.add_argument(
        "-ffsw", "--ff_smoothing_window",
        help=textwrap.dedent('''\
            Max distance in bp for fetal fraction smoothing around each SNP.
            Detail: for each SNP, the fetal fraction is smoothed by using
            the weighted averaged fetal fraction of all SNPs in a window of
            radius `ff_smoothing_window`. The average is weighted by the
            coverage at each SNP to favor fetal fraction estimated at locus
            with high coverage.
            Default value is 50e3.'''),
        type=float, default=50e3
    )

    return parser


def custom_hai_arg_parser(parser):
    """Argument parser for command line tools depending on fetal allele origin
    inference (heuristic version)

    Input:
        parser (argparse.ArgumentParser): argument parser.

    Output:
        parser (argparse.ArgumentParser): filled argument parser.
    """
    parser.add_argument(
        "-rec", "--recombination_rate", type=float,
        help=textwrap.dedent('''\
            recombination rate used in the model. If 0, recombination is
            not accounted for. Default value is 1.2e-8.'''),
        default=1.2e-8)
    parser.add_argument(
        "-mgd", "--max_genetic_dist", type=float,
        help=textwrap.dedent('''\
            Max genetic distance to define a locus neighborhood (for ambiguous
            loci) and infer (heuristic version) allele origin based on a vote
            of unambiguous loci around the considered locus. Default value
            is 1cM (centi-Morgan), i.e ~850 kbp with default recombination
            rate..'''),
        default=1e-2)

    return parser


def custom_print(string, verbose=False):
    """custom print function with verbosity control

    Input:
        string (string): string to print.
        verbose (bool): if False, no print.
    """
    if verbose:
        print(string)


def load_data(args):
    """Load sequencing data"""
    seq_data_tab = load_vcf_data(
        args.maternal_vcf, args.paternal_vcf, args.cfdna_vcf,
        region = args.region, filename = None, snp_list_file = args.snp_list,
        correct_genotype = True,
        min_rel_depth = args.min_rel_depth,
        min_abs_depth = args.min_abs_depth,
        verbose = False
    )
    # seq_data_tab = loci_tab_filter(
    #     seq_data_tab, min_coverage = args.min_coverage, verbose = args.verbose
    # )
    return seq_data_tab

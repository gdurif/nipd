#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

"""Comand line tool to infer fetal allele origin in Noninvasive Prenatal Diagnosis (NIPD)
"""

# external
import argparse
from functools import partial
import pandas as pds
import psutil
import textwrap
# internal
from prediag.bayesian_fetal_allele_origin import infer_parental_allele_origin
import prediag.cli.fetal_allele_origin_heuristic as heuristic
from prediag.cli.utils import custom_arg_parser, custom_ff_arg_parser, custom_hai_arg_parser, custom_print, load_data
from prediag.utils import float2string

def run(seq_data_tab, args):
    """Run fetal allele origin inference (heuristic version)"""
    # fetal allele origin inference (heuristic version)
    init_allele_origin_tab = heuristic.run(
        seq_data_tab, args
    )
    # fetal allele origin inference (bayesian version)
    n_gibbs = args.ncore if args.ncore > 0 else psutil.cpu_count(logical=False)
    lag = 10
    n_sample = args.nsample
    n_iter = int(n_sample * args.sampling_lag / n_gibbs)
    output_tab = infer_parental_allele_origin(
        seq_data_tab, init_allele_origin_tab,
        n_burn = args.nburn, n_iter = n_iter, lag = args.sampling_lag,
        n_thread = args.ncore, n_gibbs = n_gibbs,
        recombination_rate = args.recombination_rate,
        both_parent_phased = args.both_parent_phased,
        verbose = args.verbose, filename = args.output
    )

    # output
    return output_tab


def main():
    """Noninvasive Prenatal Diagnosis (NIPD): bayesian inference of fetal
    allele origin (parental haplotype) based on parental and cfDNA genotypes.
    """
    ## args
    parser = argparse.ArgumentParser(
        prog="prediag_fetal_allele_origin_bayesian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description = __doc__,
        epilog=textwrap.dedent('''\
            **Note:** TODO
            Example: TODO'''))
    parser = custom_arg_parser(parser)
    parser = custom_ff_arg_parser(parser)
    parser = custom_hai_arg_parser(parser)

    parser.add_argument(
        "-nc", "--ncore", type=int,
        help=textwrap.dedent('''\
            number of cores for parallel computing. Default is 1 (sequential
            computing). If 0, all available (physical) cores are used.
            If hyper-threading is activated, we do not recommend to use
            all logical cores because of over-head issues.'''),
        default=1
    )
    parser.add_argument(
        "-ns", "--nsample", type=int,
        help=textwrap.dedent('''\
            number of samples to be simulated by the Gibbs sampler.
            Default is 2000. If not large enough, inference results will not be
            trustworthy.'''),
        default=2000
    )
    parser.add_argument(
        "-nb", "--nburn", type=int,
        help=textwrap.dedent('''\
            number of preliminary burning iterations for the Gibbs Sampler,
            after which the sampling phase begins.
            Default is 2000. Should be large enough to be sure to sample under
            the posterior.'''),
        default=2000
    )
    parser.add_argument(
        "-sl", "--sampling_lag", type=int,
        help=textwrap.dedent('''\
            lag for sampling with the Gibbs sampler, i.e. number of burning
            iterations between sampling (during the sampling phase).
            Default is 100. Should be large enough to avoid correlation
            between successive samples.'''),
        default=100
    )
    parser.add_argument(
        "-bpp", "--both_parent_phased",
        help=textwrap.dedent('''\
            If enabled, the analysis is restricted to loci where
            both parents genotypes are phased (recommended).
            If disabled, all loci where at least one parent is phased
            are considered.'''),
        action="store_true"
    )
    args = parser.parse_args()

    # set verbosity
    pprint = partial(custom_print, verbose=args.verbose)

    # over
    pprint("# Starting inference")

    # load data
    seq_data_tab = load_data(args)

    # allele origin inference
    output_tab = run(seq_data_tab, args)

    # print results
    pprint(
        output_tab.dropna().to_string(
            float_format = float2string
        )
    )

    # over
    pprint("# Inference is done")


if __name__ == "__main__":
    main()

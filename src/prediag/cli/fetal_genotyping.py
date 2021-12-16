#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

"""Comand line tool for fetal genotyping in Noninvasive Prenatal Diagnosis (NIPD)
"""

# external
import argparse
from functools import partial
import textwrap
# internal
import prediag.cli.fetal_fraction_estimation
from prediag.cli.utils import custom_arg_parser, custom_ff_arg_parser, custom_print, load_data
from prediag.fetal_genotype import infer_global_fetal_genotype
from prediag.utils import float2string


def run(seq_data_tab, args):
    """Run fetal genotype inference"""
    # fetal fraction
    fetal_fraction_tab = prediag.cli.fetal_fraction_estimation.load_or_run(
        seq_data_tab, args
    )
    # fetal genotype
    fetal_genotype_tab = infer_global_fetal_genotype(
        seq_data_tab, fetal_fraction_tab.dropna(), min_coverage = args.min_coverage,
        tol = 0.0001, snp_neighborhood = args.ff_smoothing_window,
        n_neighbor_snp = 10, return_log = False, verbose = False
    )

    # output
    return fetal_genotype_tab


def main():
    """Noninvasive Prenatal Diagnosis (NIPD): inference of fetal genotype
    based on parental and cfDNA genotypes.
    """
    ## args
    parser = argparse.ArgumentParser(
        prog="prediag_fetal_genotyping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description = __doc__,
        epilog=textwrap.dedent('''\
            **Note:** TODO
            Example: TODO'''))
    parser = custom_arg_parser(parser)
    parser = custom_ff_arg_parser(parser)
    args = parser.parse_args()

    # set verbosity
    pprint = partial(custom_print, verbose=args.verbose)

    # over
    pprint("# Starting inference")

    # load data
    seq_data_tab = load_data(args)

    # fetal genotyping
    fetal_genotype_tab = run(seq_data_tab, args)

    # print results
    pprint(
        fetal_genotype_tab.dropna().to_string(
            float_format = float2string,
            formatters = {'fetal_gt_posterior': float2string}
        )
    )

    # save output
    if args.output is not None:
        try:
            fetal_genotype_tab.to_csv(args.output, index=False, sep=";")
        except BaseException:
            raise ValueError("'output' argument is not a valid file name")

    # over
    pprint("# Inference is done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

"""Comand line tool to infer fetal allele origin in Noninvasive Prenatal Diagnosis (NIPD)
"""

# external
import argparse
from functools import partial
import pandas as pds
import textwrap
# internal
import prediag.cli.fetal_genotyping
from prediag.heuristic_fetal_allele_origin import infer_parental_allele_origin
from prediag.cli.utils import custom_arg_parser, custom_ff_arg_parser, custom_hai_arg_parser, custom_print, load_data
from prediag.utils import float2string


def run(seq_data_tab, args):
    """Run fetal allele origin inference (heuristic version)"""
    # fetal fraction estimation (if required) + fetal genotyping
    fetal_genotype_tab = prediag.cli.fetal_genotyping.run(
        seq_data_tab, args
    )
    # fetal allele origin inference (heuristic version)
    output_tab = infer_parental_allele_origin(
        fetal_genotype_tab, recombination_rate = args.recombination_rate,
        genetic_dist_threshold = args.max_genetic_dist, verbose = False
    )

    # output
    return output_tab


def main():
    """Noninvasive Prenatal Diagnosis (NIPD): inference of fetal allele origin
    (parental haplotype) based on parental and cfDNA genotypes.
    """
    ## args
    parser = argparse.ArgumentParser(
        prog="prediag_fetal_allele_origin_heuristic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description = __doc__,
        epilog=textwrap.dedent('''\
            **Note:** TODO
            Example: TODO'''))
    parser = custom_arg_parser(parser)
    parser = custom_ff_arg_parser(parser)
    parser = custom_hai_arg_parser(parser)
    args = parser.parse_args()

    # set verbosity
    pprint = partial(custom_print, verbose=args.verbose)

    # over
    pprint("# Starting inference")

    # load data
    seq_data_tab = load_data(args)

    # fetal allele origin inference (heuristic version)
    output_tab = run(seq_data_tab, args)

    # print results
    pprint(
        output_tab.dropna().to_string(
            float_format = float2string,
            formatters = {'allele_origin_conf': float2string,
                          'fetal_gt_posterior': float2string}
        )
    )

    # save output
    if args.output is not None:
        try:
            output_tab.to_csv(args.output, index=False, sep=";")
        except BaseException:
            raise ValueError("'output' argument is not a valid file name")


    # over
    pprint("# Inference is done")


if __name__ == "__main__":
    main()

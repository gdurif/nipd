#!/usr/bin/env python

# This file is part of the `prediag` package which is released under GPL-v3.
# See the attached files LICENSE.txt and COPYING.txt for full license details.

"""Comand line tool for fetal fraction estimation in Noninvasive Prenatal Diagnosis (NIPD)
"""

# external
import argparse
from functools import partial
import pandas as pd
import textwrap
from warnings import warn as warning
# internal
from prediag.cli.utils import custom_arg_parser, custom_print, load_data
from prediag.fetal_fraction import estimate_global_fetal_fraction, impute_fetal_fraction
from prediag.utils import float2string


def run(seq_data_tab, args):
    """Run fetal fraction estimation"""
    # fetal fraction estimation
    fetal_fraction_tab = estimate_global_fetal_fraction(
        seq_data_tab, min_coverage = args.min_coverage, tol = 0.05
    )

    # # missing fetal fraction inference (useless due to smoothing?)
    # fetal_fraction_tab = impute_fetal_fraction(fetal_fraction_tab)

    # output
    return fetal_fraction_tab


def load_or_run(seq_data_tab, args):
    """Load fetal fraction result or run fetal fraction estimation if missing"""
    # FIXME check CSV file formating
    if args.fetal_fraction_file is None:
        return run(seq_data_tab, args)
    else:
        return pd.read_csv(args.fetal_fraction_file, sep = ";")


def main():
    """Noninvasive Prenatal Diagnosis (NIPD): estimation of fetal fraction
    in cfDNA sequencing data.
    """
    ## args
    parser = argparse.ArgumentParser(
        prog="prediag_fetal_fraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description = __doc__,
        epilog=textwrap.dedent('''\
            **Note:** TODO
            Example: TODO'''))
    parser = custom_arg_parser(parser)
    args = parser.parse_args()

    # set verbosity
    pprint = partial(custom_print, verbose=args.verbose)

    # over
    pprint("# Starting estimation procedure")

    # load data
    seq_data_tab = load_data(args)

    # fetal fraction estimation
    fetal_fraction_tab = run(seq_data_tab, args)

    # print results
    pprint(fetal_fraction_tab.dropna().to_string(float_format = float2string))

    # save output
    if args.output is not None:
        if len(fetal_fraction_tab.dropna().index) > 0:
            try:
                fetal_fraction_tab.dropna().to_csv(args.output, index=False, sep=";")
            except BaseException as exc:
                raise exc
        else:
            text = "Fetal fraction cannot be estimated for any locus. No table was saved."
            warning(text)


    # over
    pprint("# Estimation is done")


if __name__ == "__main__":
    main()

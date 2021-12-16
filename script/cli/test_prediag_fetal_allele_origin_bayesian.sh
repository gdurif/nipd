#!/bin/bash

## VCF files
MATERNAL_VCF=
PATERNAL_VCF=
CFDNA_VCF=

## run
prediag_fetal_allele_origin_bayesian \
-cfdna_vcf $CFDNA_VCF \
-mat_vcf $MATERNAL_VCF \
-pat_vcf $PATERNAL_VCF \
--output allele_origin.csv \
-r "chr4:3000000:3200000" \
--min_coverage 50 \
--min_rel_depth 0.02 \
--min_abs_depth 2 \
--ff_smoothing_window 50e3 \
--fetal_fraction_file fetal_fraction.csv \
--recombination_rate 1e-8 \
--max_genetic_dist 1e-2 \
--ncore 0 \
--nsample 500 \
--nburn 200 \
--sampling_lag 5 \
--both_parent_phased \
-v

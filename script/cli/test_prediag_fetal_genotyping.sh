#!/bin/bash

## VCF files
MATERNAL_VCF=
PATERNAL_VCF=
CFDNA_VCF=

## run
prediag_fetal_genotyping \
-cfdna_vcf $CFDNA_VCF \
-mat_vcf $MATERNAL_VCF \
-pat_vcf $PATERNAL_VCF \
-o genotype.csv \
-r "chr4:3000000:3200000" \
--min_coverage 50 \
--min_rel_depth 0.02 \
--min_abs_depth 2 \
--fetal_fraction_file fetal_fraction.csv \
--ff_smoothing_window 50e3 \
-v

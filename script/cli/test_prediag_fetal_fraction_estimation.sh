#!/bin/bash

## VCF files
MATERNAL_VCF=
PATERNAL_VCF=
CFDNA_VCF=

## run
prediag_fetal_fraction \
-cfdna_vcf $CFDNA_VCF \
-mat_vcf $MATERNAL_VCF \
-pat_vcf $PATERNAL_VCF \
-o fetal_fraction.csv \
--min_coverage 50 \
--min_rel_depth 0.02 \
--min_abs_depth 2 \
-v

#!/usr/bin/env python
## test VCF reading

# external
import matplotlib.pyplot as plt
import numpy as np
import os
# internal
from prediag.vcf_reader import load_vcf_data

## VCF files
mother_vcf = ""
father_vcf = ""
cfdna_vcf = ""

## read VCF files
df = load_vcf_data(mother_vcf, father_vcf, cfdna_vcf)

print(df.to_string())

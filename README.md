# Bayesian inference for non-invasive prenatal diagnostic of genetic diseases

Bayesian approach to infer the fetal fraction, the fetal genotype and the fetal allele origin using sequencing data from the mother, the father and cfDNA in the maternal plasma [1].

## Reference

[1] C. Liautard-Haag, G. Durif, C. Vangoethem, D. Baux, A. Louis, et al.. Noninvasive prenatal diagnosis of genetic diseases induced by triplet repeat expansion by linked read haplotyping and Bayesian approach. Scientific Reports, 2022, 12 (1), pp.11423. [⟨10.1038/s41598-022-15307-2⟩](https://dx.doi.org/10.1038/s41598-022-15307-2). [⟨hal-03716132⟩](https://hal.archives-ouvertes.fr/hal-03716132).

## Authors

- Ghislain DURIF [a]
- Cathy LIAUTARD HAAG [b]
- Marie-Claire VINCENT [b,c]

[a] IMAG, Université de Montpellier, CNRS, Montpellier, France;

[b] Laboratoire de Génétique Moléculaire, Institut Universitaire de Recherche Clinique, Université de Montpellier, CHU Montpellier, Montpellier, France;

[c] PhyMedExp Univ. Montpellier, CNRS, INSERM, Montpellier, France;

## Availability

The pipeline is available as a Python package called `prediag` (see the [`src`](./src) directory), which provides a command line interface (CLI) tools. See below for instructions about [installation](#installation) and [usage](#usage).

## Licensing

The `prediag` package is released under the GPL-v3 license. See the attached files [`LICENSE.txt`](./LICENSE.txt) and [`COPYING.txt`](./COPYING.txt) for full license details.

---

## Installation

### Requirements

External software:
- Python 3+
- git (optional)

> **Note:** To avoid messing with your system, we recommend to install the `prediag` Python package inside a Python virtual environment or inside a Conda environment (c.f. [below](#using-a-python-environment) for more details).

### Install from remote sources

You can install the `prediag` package from the remote repository using `pip` with:
```bash
pip install git+https://github.com/gdurif/nipd.git#subdirectory=src
```

### Install from local sources

* Get the package sources
```bash
git clone https://github.com/gdurif/nipd
```

* Installation
```bash
# go to source directory
cd nipd/src
# install
pip install -e .
```

---

## Usage

> Before using the package (which provides command line tools), if relevant, you should **activate** your Python or Conda environment (depending on which you are using, c.f. [below](#using-a-python-environment)).

### Command line interface (CLI)

You can find bash script examples to use the `prediag` command line interface in ths dedicated [`script/cli`](./script/cli) folder.

Some arguments/options are optional, c.f. help (available with the `-h` options).

#### Fetal fraction estimation

* Command line tool help
```bash
prediag_fetal_fraction -h
```

* Using command line tool for fetal fraction estimation
```bash
prediag_fetal_fraction \
-cfdna_vcf cfnda_vcf_file \
-mat_vcf mother_vcf_file \
-pat_vcf father_vcf_file \
--output ff_output_file.csv \
--min_coverage 50 \
--min_rel_depth 0.02 \
--min_abs_depth 2 \
-v
```

#### Fetal genotyping

* Command line tool help
```bash
prediag_genotyping -h
```

* Using command line tool for fetal genotype inference on a given region (being `chr4:3000000:3200000`):
```bash
prediag_fetal_genotyping \
-cfdna_vcf cfnda_vcf_file \
-mat_vcf mother_vcf_file \
-pat_vcf father_vcf_file \
--output gt_output_file.csv \
--region "chr4:3000000:3200000" \
--min_coverage 50 \
--min_rel_depth 0.02 \
--min_abs_depth 2 \
--fetal_fraction_file ff_output_file.csv \
--ff_smoothing_window 50e3 \
-v
```

> **Note**: you can provide the output of fetal fraction estimation (potentially on a wider set of loci) with the option `--fetal_fraction_file`.


#### Fetal allele origin inference

* Command line tool help
```bash
prediag_fetal_allele_origin_bayesian -h
```

* Using command line tool for fetal allele origin inference on a given region (being `chr4:3000000:3200000`):
```bash
prediag_fetal_allele_origin_bayesian \
-cfdna_vcf cfnda_vcf_file \
-mat_vcf mother_vcf_file \
-pat_vcf father_vcf_file \
--output allele_origin_output_file.csv \
--region "chr4:3000000:3200000" \
--min_coverage 50 \
--min_rel_depth 0.02 \
--min_abs_depth 2 \
--ff_smoothing_window 50e3 \
--fetal_fraction_file ff_output_file.csv \
--recombination_rate 1e-8 \
--max_genetic_dist 1e-2 \
--ncore 0 \
--nsample 5000 \
--nburn 2000 \
--sampling_lag 50 \
--both_parent_phased \
-v
```

> **Note**: you can provide the output of fetal fraction estimation (potentially on a wider set of loci) with the option `--fetal_fraction_file`.


### Python package

You can find Python script examples to use the `prediag` Python package in a dedicated [`script/python`](./script/python) folder.

---

## Changelog

### v1.0.1

- allow to provide fetal fraction estimation computed on wider region to fetal genotyping and fetal allele origin inference
- filter loci by minimum coverage
- correct cfDNA genotype by checking that allelic depth is higher than an absolute threshold, and higher than a percentage of the coverage
- window-based estimation of fetal fraction with a weighted average of locus-based fetal fraction estimation with weights proportional to coverage
- account for chromosome break in fetal allele origin inference

---

## Using a Python environment

To avoid messing with your system, we recommend that you install the `prediag` Python package inside a Python virtual environment or inside a Conda environment.

### Python virtual environment

* Create Python virtual environment
```bash
# edit the path
python -m venv /path/to/.pyenv
```

* Activate the Python virtual environment (**should be done before every (re)installation/use of the package**):
```bash
# edit the path
source /path/to/.pyenv/bin/activate
```

### Conda environment

* You should install Anaconda or Miniconda Python distribution. To check if it is available, run:
```bash
conda list
```

* Create a Conda environment:
```bash
# replace `<name_of_the_environment>` by a name of your choice
conda create -n <name_of_the_environment>
```

* Activate the Conda environment (**should be done before every (re)installAtion/use of the package**):
```bash
# replace `<name_of_the_environment>` by the name you chose
conda activate <name_of_the_environment>
```

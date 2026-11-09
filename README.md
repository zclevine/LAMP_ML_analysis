# LAMP_ML_analysis

Code for designing LAMP (loop-mediated isothermal amplification) primer sets, analyzing high-throughput LAMP data, and calculating thermodynamic properties of primer set-target pairs. Includes primer design and scoring, generation of wild-type and mutant target sequences, and analysis of BioMarkX9 (Fluidigm) fluorescence runs. Also includes source code for LAMP-ADAPT, a machine-learning enabled tool for LAMP primer set design which predicts amplification speed and scores primers based on speed and inclusivity. 

## Repository contents

### Primer design and scoring
| File | Description |
|---|---|
| `primer_design_class.ipynb` | Primer design for a target sequence: generates all primers and primer sets, checks Tm, length, amplicon and loop characteristics, minimum within-set ΔG, and mismatches to targets, and builds primer bind and ordering files. |
| `calculate_bindings.py` | Helpers for primer binding: degenerate-base-aware mismatch counting and thermodynamic binding calculations (primer3) at LAMP conditions (65 °C, 50 mM Na⁺, 8 mM Mg²⁺, 1.4 mM dNTPs, 100 nM primer). |
| `find_primer_hits.py` | Locates primers (F3, F2, F1c, LF, B1c, B2, B3, LB) on consensus sequences or alignments, allowing a set number of mismatches and checking orientation. |
| `parse_GLAPD_primers.py` | Parses primer set output from GLAPD into tidy pandas DataFrames (primers and genomes per set). |

### Target sequence generation
| File | Description |
|---|---|
| `DinucleotideAnalysis` | `DinucleotideAnalysis` class: scaling/CLR transformation and PCA of dinucleotide frequencies. | Adapted from Metsky et al https://doi.org/10.1038/s41587-022-01213-5
| `WT_target_generation` | `TargetGeneration` class: samples new target sequences from dinucleotide PCA space, built from the PCA above. |
| `LAMP_ML_mutant_target_design.ipynb` | Design of mutant targets. |

### BioMark run analysis
| File | Description |
|---|---|
| `BioMarkAnalysisLML_v20260522.py` | Current `BioMarkAnalysisLML` class. Converts a 96-well plate layout to the BioMark IFC chip layout, processes raw FAM/ROX fluorescence, and writes figures and results to a dated output folder. |
| `BioMarkAnalysisLASV_FINAL.py` | Final `BioMarkAnalysisLASV` class used for the Lassa virus (LASV) experiments. |

### LAMP-ADAPT
Self-contained code for LAMP-ADAPT predictions (see README.md in folder) 

See the notebooks for end-to-end examples.

## Data

Raw data is not included in this repository. Study data is available at 10.5281/zenodo.22901592

## Citation

Manuscript in preparation; contact for details

## Contact

Zoë Levine, Sabeti Lab, Broad Institute (zlevine@broadinstitute.org)
Tsion Abay, Sabeti Lab, Broad Institute (tabay@broadinstitute.org)
Sameed Siddiqui, Sabeti Lab, Broad Institute (ssiddiqui@broadinstitute.org)
Pardis Sabeti, Sabeti Lab, Broad Institute (pardis@broadinstitute.org)# LAMP_ML

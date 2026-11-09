import subprocess
import re

import numpy as np
import pandas as pd
import primer3
from Bio.SeqUtils import gc_fraction
from typing import Optional, Tuple, List


DNA_COMP = str.maketrans("ACGTacgt", "TGCAtgca")

def revcomp(seq: str) -> str:
	return seq.translate(DNA_COMP)[::-1]


def count_mismatches(seq1, seq2, blast_result=False, include_base=False):
	"""Count mismatches between seq1 and seq2, respecting degenerate bases."""
	degenerate_map = {
		'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'},
		'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'S': {'G', 'C'},
		'W': {'A', 'T'}, 'K': {'G', 'T'}, 'M': {'A', 'C'},
		'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'}, 'H': {'A', 'C', 'T'},
		'V': {'A', 'C', 'G'}, 'N': {'A', 'C', 'G', 'T'}
	}

	if len(seq1) != len(seq2):
		raise ValueError("Sequences are of unequal lengths.")

	mismatches = 0
	mismatch_positions = []

	for i, (base1, base2) in enumerate(zip(seq1, seq2)):
		set1 = degenerate_map.get(base1, {base1})
		set2 = degenerate_map.get(base2, {base2})
		if set1.isdisjoint(set2):
			mismatches += 1
			if blast_result:
				if include_base:
					mismatch_positions.append(f"{base1}{i}{base2}")
				else:
					mismatch_positions.append(i + 1)

	if blast_result:
		return mismatches, mismatch_positions
	else:
		return mismatches


def calc_and_reshape_bindings(
	target_df: pd.DataFrame,
	primer_types,
	*,
	na_conc: float = 50.0,	   # mM monovalent cations
	mg_conc: float = 8.0,	   # mM divalent cations
	oligo_conc: float = 100.0, # nM primer concentration (0.1 µM)
	dntp_conc: float = 1.4,	   # mM dNTPs
	temp: float = 65.0,		   # °C
	blast_result = False,
	include_base = False,
	RNAduplex_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Compute binding metrics for primer/target pairs (long format),
	then reshape to wide format by (primer_set, target_name, primer_type).

	target_df must have (at least) columns:
	  - 'primer_set'
	  - 'primer_type'
	  - 'primer_name'
	  - 'primer_sequence'
	  - 'target_name'
	  - 'target_sequence'

	Parameters
	----------
	target_df : pandas.DataFrame
		Long-format DataFrame with one row per primer/target pair.
	primer_types : list of str
		List of primer types (e.g. ['F3','F2','F1c','LF','B3','B2','B1c','LB']).
	na_conc, mg_conc, oligo_conc, dntp_conc, temp : float
		Thermodynamic parameters passed to primer3.
	RNAduplex_path : str or None
		If provided, path to RNAduplex binary. If None, dg_ViennaRNA is NaN.

	Returns
	-------
	bindings_long : pandas.DataFrame
		target_df with extra columns:
		  - dg_ViennaRNA
		  - gc_fraction
		  - dg_primer3
		  - mismatches
		  - tm

	bindings_wide : pandas.DataFrame
		Index: (primer_set, target_name)
		For each primer_type p, columns:
		  - f'{p}'					   (primer_name)
		  - f'{p}_sequence'			   (primer_sequence)
		  - f'{p}_target_sequence'	   (target_sequence)
		  - f'{p}_dg_ViennaRNA'
		  - f'{p}_dg_primer3'
		  - f'{p}_mismatches'
		  - f'{p}_tm'
		  - f'{p}_gc_fraction'
	"""
	# ---- 1. LONG-FORMAT: compute bindings ----
	df = target_df.copy()

	# unique primer/target seq pairs for speed
	unique_pairs = df[["primer_sequence", "target_sequence","primer_type"]].drop_duplicates()

	# ViennaRNA ΔG (optional)
	if RNAduplex_path is not None:
		def get_deltaG_vienna(seq1, seq2):
			inp = f"{seq1}\n{seq2}\n"
			res = subprocess.run(
				[RNAduplex_path, "--noconv", "--paramFile=DNA"],
				input=inp,
				capture_output=True,
				text=True,
				check=True
			)
			deltaG = re.search(r"\(\s*([-+]?\d*\.?\d+)\s*\)", res.stdout)
			if not deltaG:
				raise ValueError(
					f"Could not parse ΔG from RNAduplex output:\n{res.stdout}"
				)
			return float(deltaG.group(1))

		unique_pairs["dg_ViennaRNA"] = unique_pairs.apply(
			lambda row: get_deltaG_vienna(
				row["primer_sequence"],
				(revcomp(row["target_sequence"]))
			),
			axis=1
		)
	else:
		unique_pairs["dg_ViennaRNA"] = np.nan

	# GC fraction of primer
	unique_pairs["gc_fraction"] = unique_pairs["primer_sequence"].apply(gc_fraction)


	# primer3 heterodimer ΔG (primer vs rc(target))
	unique_pairs["dg_primer3"] = unique_pairs.apply(
		lambda row: primer3.calc_heterodimer(
			row["primer_sequence"],
			revcomp(row["target_sequence"]),
			mv_conc=na_conc,
			dv_conc=mg_conc,
			dntp_conc=dntp_conc,
			dna_conc=oligo_conc,
			temp_c=temp
		).dg,
		axis=1
	)

	# primer3 end stability 
	# end stability is calculated relative to the direction of the extension
	# ie for F1c and B1c end_stability(rev_comp(primer), target) 
	# for all other primer types end_stability(primer, rev_comp(target))
	def calc_end_stability_lamp(row):
		if row["primer_type"] in ("F1c", "B1c"):
			s1 = revcomp(row["primer_sequence"])
			s2 = row["target_sequence"]	 
		else:
			s1 = row["primer_sequence"]
			s2 = revcomp(row["target_sequence"])
		
		return primer3.calc_end_stability(
			s1, s2,
			mv_conc=na_conc,
			dv_conc=mg_conc,
			dntp_conc=dntp_conc,
			dna_conc=oligo_conc,
			temp_c=temp
		).dg

	unique_pairs["end_stability_primer3"] = unique_pairs.apply(calc_end_stability_lamp, axis=1)

	# mismatches (primer vs target)
	unique_pairs["mismatches"] = unique_pairs.apply(
		lambda row: count_mismatches(
			row["primer_sequence"],
			row["target_sequence"]
		),
		axis=1
	)

	# primer Tm
	unique_pairs["tm"] = unique_pairs["primer_sequence"].apply(
		lambda s: primer3.calc_tm(
			s,
			mv_conc=na_conc,
			dv_conc=mg_conc,
			dntp_conc=dntp_conc,
			dna_conc=oligo_conc,
			tm_method="santalucia",
			salt_corrections_method="santalucia"
		)
	)

	# Merge metrics back to full target_df (long-format)
	bindings_long = df.merge(
		unique_pairs,
		on=["primer_sequence", "target_sequence", "primer_type"],
		how="left"
	)

	# ---- 2. WIDE-FORMAT: reshape ----
	required_cols = {
		"primer_set", "target_name", "primer_type",
		"primer_name", "primer_sequence", "target_sequence",
		"dg_ViennaRNA", "dg_primer3",
		"mismatches", "tm", "gc_fraction"
	}
	missing = required_cols - set(bindings_long.columns)
	if missing:
		raise ValueError(f"bindings_long is missing required columns: {missing}")

	base_index_cols = ["primer_set", "target_name"]

	df_w = bindings_long.set_index(base_index_cols + ["primer_type"], drop=False)
	all_index = df_w.index.droplevel("primer_type").drop_duplicates()
	bindings_wide = pd.DataFrame(index=all_index)

	metric_cols = [
		"dg_ViennaRNA",
		"dg_primer3",
		"end_stability_primer3",
		"mismatches",
		"tm",
		"gc_fraction",
	]

	for ptype in primer_types:
		df_p = df_w[df_w["primer_type"] == ptype]

		if df_p.empty:
			cols = {
				ptype: pd.Series(index=bindings_wide.index, dtype=object),
				f"{ptype}_sequence": pd.Series(index=bindings_wide.index, dtype=object),
				f"{ptype}_target_sequence": pd.Series(index=bindings_wide.index, dtype=object),
			}
			for mc in metric_cols:
				cols[f"{ptype}_{mc}"] = pd.Series(index=bindings_wide.index, dtype=float)
			df_p_wide = pd.DataFrame(cols)
		else:
			df_p = df_p.reset_index(drop=True)
			df_p = df_p.drop_duplicates(
				subset=["primer_set", "target_name", "primer_type"]
			)
			df_p = df_p.set_index(base_index_cols, drop=True)

			data = {
				ptype: df_p["primer_name"],
				f"{ptype}_sequence": df_p["primer_sequence"],
				f"{ptype}_target_sequence": df_p["target_sequence"],
			}
			for mc in metric_cols:
				data[f"{ptype}_{mc}"] = df_p[mc]

			df_p_wide = pd.DataFrame(data)

		bindings_wide = bindings_wide.join(df_p_wide, how="outer")

	bindings_wide = bindings_wide.sort_index()
	bindings_wide['total_mismatches'] = bindings_wide[[f'{ptype}_mismatches' for ptype in primer_types]].sum(axis=1)
	bindings_wide['mean_dg_ViennaRNA'] = bindings_wide[[f'{ptype}_dg_ViennaRNA' for ptype in primer_types]].mean(axis=1)
	bindings_wide['mean_dg_primer3'] = bindings_wide[[f'{ptype}_dg_primer3' for ptype in primer_types]].mean(axis=1)
	bindings_wide['mean_end_stability_primer3'] = bindings_wide[[f'{ptype}_end_stability_primer3' for ptype in primer_types]].mean(axis=1)
	return bindings_long, bindings_wide
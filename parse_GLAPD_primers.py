#!/usr/bin/env python3

import re
import pandas as pd

def parse_GLAPD_primers_to_dfs(input_file):
    """
    Parse LAMP primers from text file.

    Returns
    -------
    primers_df : pandas.DataFrame
        Columns:
            - primer_set
            - primer_type
            - primer_name
            - sequence   (NaN if primer is NULL)

    genomes_df : pandas.DataFrame
        Columns:
            - primer_set
            - num_genomes
            - genomes
    """
    primer_rows = []   # one row per primer
    genome_rows = []   # one row per primer_set

    with open(input_file, 'r') as f:
        content = f.read()

    # Split by primer set headers, keeping the header tokens
    sets = re.split(r'(The \d+ LAMP primers:)', content)

    for i in range(1, len(sets), 2):
        if i + 1 >= len(sets):
            continue

        header = sets[i]
        body = sets[i + 1]

        # Extract set number
        set_num_match = re.search(r'The (\d+) LAMP primers:', header)
        if not set_num_match:
            continue

        set_num = set_num_match.group(1)
        primer_set_id = f'PS{set_num}'

        # ---- Parse primers for this set ----
        primer_lines = body.split('\n')
        for line in primer_lines:
            line = line.strip()
            if not line:
                continue

            # NULL primer: "<name>: NULL"
            null_match = re.match(r'([^:]+):\s*NULL', line)
            if null_match:
                primer_name = null_match.group(1).strip()
                primer_type = primer_name  # using name as type
                primer_rows.append({
                    'primer_set': primer_set_id,
                    'primer_type': primer_type,
                    'primer_name': primer_name,
                    'sequence': None   # becomes NaN in pandas
                })
                continue

            # Regular primers with pos and sequence:
            # "<name>: pos:123,length:20 bp, primer(5'-3'):ACGT..."
            if 'pos:' in line:
                primer_match = re.match(
                    r'([^:]+):\s*pos:(\d+),length:(\d+)\s*bp,\s*primer\(5\'-3\'\):(\w+)',
                    line
                )
                if primer_match:
                    primer_type = primer_match.group(1).strip()
                    sequence = primer_match.group(4)
                    primer_name = primer_set_id + "_" + primer_type

                    primer_rows.append({
                        'primer_set': primer_set_id,
                        'primer_type': primer_type,
                        'primer_name': primer_name,
                        'sequence': sequence
                    })

        # ---- Parse genome information for this set ----
        set_record = {'primer_set': primer_set_id}

        genome_num_match = re.search(r'could be used in (\d+) genomes?', body)
        if genome_num_match:
            set_record['num_genomes'] = genome_num_match.group(1)
        else:
            set_record['num_genomes'] = ''

        genome_match = re.search(
            r'there are:\s*(.+?)(?=The \d+ LAMP primers:|$)',
            body,
            re.DOTALL
        )
        if genome_match:
            genomes = genome_match.group(1).strip()
            genomes = ' '.join(genomes.split())
            set_record['genomes'] = genomes
        else:
            set_record['genomes'] = ''

        genome_rows.append(set_record)

    primers_df = pd.DataFrame(
        primer_rows,
        columns=['primer_set', 'primer_type', 'primer_name', 'sequence']
    )
    genomes_df = pd.DataFrame(
        genome_rows,
        columns=['primer_set', 'num_genomes', 'genomes']
    )

    return primers_df, genomes_df
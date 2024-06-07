# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %%
import argparse

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input file')
    parser.add_argument('output', type=str, help='output file')
    args = parser.parse_args()
    with open(args.input) as f:
        lines = f.readlines()
    with open(args.output, 'w') as f:
        for line in lines[1:]:
            f.write(' '.join(line.strip().split(',')[1:]) + '\n')

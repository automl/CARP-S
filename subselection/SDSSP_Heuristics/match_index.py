#!/usr/bin/env python
# coding: utf-8
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
from decimal import Decimal
import argparse


# %%
def build_list(values):
    result = {}
    for i, v in enumerate(values):
        if v not in result:
            result[v] = []
        result[v].append(i)
    return result


# %%
def distance_inf(point1, point2):
    ans = 0
    point = (point1[0], point2[0])
    assert len(point1) == len(point2)
    for i in range(len(point1)):
        new_ans = abs(point1[i] - point2[i])
        if new_ans > ans:
            ans = new_ans
            point = (point1[i], point2[i])
    return ans, point

def isclose(a, b, rel_tol=1e-15, abs_tol=0.0):
    ans = abs(a-b) <= max(Decimal(rel_tol) * max(abs(a), abs(b)), abs_tol)
    return ans

# %%
def match_index(index_list, values):
    pointers = [0] * len(index_list)
    results = []
    for v in values:
        found = 0
        found_l = []
        for i, w in enumerate(index_list.keys()):
            if isclose(*distance_inf(v, w)[1]):
                if found < 1:
                    results.append(index_list[w][pointers[i]])
                    pointers[i] += 1
                found_l.append(list(map(str, w)))
                found += 1
        if found > 1:
            print(found)
            for vv in found_l:
                print(*vv)
            print(*list(map(str, v)))
            print("Warning: Multiple matches found")
            raise ValueError("Multiple matches found")
        if found == 0:
            raise ValueError("No match found")
    return results


# %%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('origin_csv', type=str)
    parser.add_argument('remaining_points', type=str)
    parser.add_argument('output_csv', type=str)
    parser.add_argument('complement_csv', type=str)
    args = parser.parse_args()
    with open(args.origin_csv, 'r') as f:
        origin_csv = f.readlines()
    with open(args.remaining_points, 'r') as f:
        remaining_points = f.readlines()
    index_list = build_list([tuple(map(Decimal, x.strip().split(',')[1:])) for x in origin_csv[1:]])
    ans = match_index(index_list, [tuple(map(Decimal, x.split())) for x in remaining_points[1:]])
    with open(args.output_csv, 'w') as f:
        f.write(origin_csv[0])
        for i in ans:
            f.write(origin_csv[i+1])
    
    with open(args.complement_csv, 'w') as f:
        f.write(origin_csv[0])
        ans = set(ans)
        for i in range(len(origin_csv)-1):
            if i not in ans:
                f.write(origin_csv[i+1])
    


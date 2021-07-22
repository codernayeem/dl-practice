from pathlib import Path
import json
import pandas as pd


EXECUTION_LINE_START = '      "execution_count":'

def find_index(lst, item):
    for i, v in enumerate(lst):
        if v == item:
            return i
    return None

def read_file(path, execuion_line=EXECUTION_LINE_START):
    p = Path(path)
    if not (p.exists() and p.is_file()):
        raise ValueError(f'File not Found : "{path}"')

    line_index, content = [], []
    for i, line in enumerate(p.read_text().split('\n')):
        if line.startswith(execuion_line):
            line_index.append(i)
            content.append(line)
    return line_index, content

def save_file(orginial, path, line_index, content):
    p = Path(orginial)
    if not (p.exists() and p.is_file()):
        raise ValueError(f'File not Found : "{orginial}"')

    r = []
    for i, line in enumerate(p.read_text().split('\n')):
        index = find_index(line_index, i)
        if index == None:
            r.append(line)
        else:
            r.append(content[index])

    with open(path, 'w') as fl:
        fl.write('\n'.join(r))

def fix_all(content):
    for i in range(len(content)):
        has_comma = content[i].endswith(',')
        start_index = content[i].index(':')
        content[i] = content[i][:start_index] + f': {i+1}'
        if has_comma:
            content[i] += ','
    return content


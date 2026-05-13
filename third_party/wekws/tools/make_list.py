#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Jing Du(thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import re


def read_token(token_file):
    token_table = {}
    with open(token_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            if len(arr) < 2:
                continue
            token_table[arr[0]] = int(arr[1])
    return token_table


def read_lexicon(lexicon_file):
    lexicon_table = {}
    with open(lexicon_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            if len(arr) < 2:
                continue
            lexicon_table[arr[0]] = arr[1:]
    return lexicon_table


def query_token_set(keyword, token_table, lexicon_table=None):
    lexicon_table = lexicon_table or {}
    pieces = []
    for unit in split_mixed_label(keyword.strip().replace(' ', '')):
        if unit in lexicon_table:
            pieces.extend(lexicon_table[unit])
        else:
            pieces.append(unit)

    indexes = []
    for token in pieces:
        if token in token_table:
            indexes.append(token_table[token])
        elif token.upper() in token_table:
            indexes.append(token_table[token.upper()])
        elif token.lower() in token_table:
            indexes.append(token_table[token.lower()])
        else:
            raise KeyError(f'token {token} was not found in token table')

    return pieces, indexes


def split_mixed_label(input_str):
    tokens = []
    s = input_str
    while len(s) > 0:
        match = re.match(r'[A-Za-z!?,<>_()\']+', s)
        if match is not None:
            word = match.group(0)
        else:
            word = s[0:1]
        tokens.append(word)
        s = s.replace(word, '', 1).strip(' ')
    return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('text_file', help='text file')
    parser.add_argument('duration_file', help='duration file')
    parser.add_argument('output_file', help='output list file')
    parser.add_argument('--token_file', default=None, help='token file')
    parser.add_argument('--lexicon_file', default=None, help='lexicon file')
    args = parser.parse_args()

    token_table = None
    lexicon_table = None
    if args.token_file is not None:
        token_table = read_token(args.token_file)
    if args.lexicon_file is not None:
        lexicon_table = read_lexicon(args.lexicon_file)

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    duration_table = {}
    with open(args.duration_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            duration_table[arr[0]] = float(arr[1])

    with open(args.text_file, 'r', encoding='utf8') as fin, \
         open(args.output_file, 'w', encoding='utf8') as fout:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            if len(arr) < 2:
                txt = '<SILENCE>'
            else:
                txt = ' '.join(split_mixed_label(arr[1]))
                if token_table is not None:
                    _, indexes = query_token_set(arr[1], token_table,
                                                 lexicon_table)
                    txt = ' '.join(str(i) for i in indexes)
            assert key in wav_table
            wav = wav_table[key]
            assert key in duration_table
            duration = duration_table[key]
            line = dict(key=key, txt=txt, duration=duration, wav=wav)

            json_line = json.dumps(line, ensure_ascii=False)
            fout.write(json_line + '\n')

#!/usr/bin/env python3

import json
import sys

from random import shuffle

if __name__ == '__main__':
    all_data = []

    for input_file in sys.argv[1:]:
        with open(input_file, "r") as f:
            this_data = json.load(f)
            all_data += this_data

    shuffle(all_data)

    json.dump(all_data, sys.stdout)
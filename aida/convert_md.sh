#!/bin/bash

set -x

if [ "$#" -eq 0 ]; then
    python utils/convert_md.py
else
    python utils/convert_md.py $1
fi


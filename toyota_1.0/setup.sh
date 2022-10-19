#!/bin/bash

# Where to store the datasets?
# mkdir -p data/toyota/preprocessed/
# mkdir -p data/openacc/preprocessed/

# Where to store the logs/models of trained models
mkdir -p results/toyota/logs/
mkdir -p results/toyota/models/

mkdir -p results/openacc/logs/
mkdir -p results/openacc/models/

mkdir -p results/once/logs/
mkdir -p results/once/models/

# echo "============= Preprocessing datasets ============="
cd code ; python preprocess.py

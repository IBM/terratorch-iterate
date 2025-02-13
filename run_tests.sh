#!/bin/bash

rm ~/test-terratorch-iterate.err
rm ~/test-terratorch-iterate.out
jbsub -e ~/test-terratorch-iterate.err -o ~/test-terratorch-iterate.out -m 20G  -c 1+1 -r v100 pytest --cov-report html --cov=benchmark tests/test_benchmark.py
#!/bin/bash

echo "Running tests"

std_err_file="~/test-terratorch-iterate.err"

# Using the test command
if [ -e "$std_err_file" ]
then
  echo "Standard Err File exists, so it will be removed to create a new one from scratch"
  rm $std_err_file
else
  echo "Standard Err File does not exist, so it will be created"
fi

std_out_file="~/test-terratorch-iterate.out"

# Using the test command
if [ -e "$std_out_file" ]
then
  echo "Standard out File exists, so it will be removed to create a new one from scratch"
  rm $std_out_file
else
  echo "Standard out File does not exist, so it will be created"
fi

echo "Submit test job"

jbsub -e ~/test-terratorch-iterate.err -o ~/test-terratorch-iterate.out -m 20G  -c 1+1 -r v100 pytest --cov-report html --cov=benchmark tests/test_benchmark.py
#!/bin/bash

echo "Running tests"

# set standard err file
STD_ERR_FILE="~/test-terratorch-iterate.err"

# Using the test command
if [ -f "$STD_ERR_FILE" ]; then
  echo "Standard Err File exists, so it will be removed to create a new one from scratch"
  rm $STD_ERR_FILE
else
  echo "Standard Err File does not exist, so it will be created"
fi

# set standard out file
STD_OUT_FILE="~/test-terratorch-iterate.out"

# Using the test command
if [ -f "$STD_OUT_FILE" ]; then
  echo "Standard out File exists, so it will be removed to create a new one from scratch"
  rm $STD_OUT_FILE
else
  echo "Standard out File does not exist, so it will be created"
fi

echo "Submit test job"

jbsub -e $STD_ERR_FILE -o $STD_OUT_FILE -m 20G  -c 1+1 -r v100 pytest --cov-report html --cov=benchmark tests/test_benchmark.py
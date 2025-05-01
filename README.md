[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
![alt text](./coverage.svg)
# TerraTorch-iterate

A tool for benchmarking and hyper-parameter optimization using [TerraTorch](https://github.ibm.com/GeoFM-Finetuning/terratorch).

Leverages MLFlow for experiment logging, optuna for hyperparameter optimization and ray for parallelization.

## Installation

We recommend using python 3.10, 3.11 or 3.12 and also using a virtual environment for all commands in this guide.

### Package installation

```sh
pip install terratorch-iterate
```

### Suggested setup for development

```sh
pip install --upgrade pip setuptools wheel
pip install -e .
pip install -e ".[dev]"
pip install -e ".[test]"
pip install -e ".[utilities]"
```

## Usage for benchmarking

This tool allows you to design a benchmark test for a backbone that exists in `TerraTorch` over:

- Several tasks

- Several hyperparameter configurations

To do this it relies on a configuration file where the benchmark is defined. This consists of:

- `experiment_name`: MLFLow experiment to run the benchmark on. This is the highest level grouping of runs in MLFLow.

- `run_name`: Name of the parent (top-level) run under the experiment. NOTE: This item should not be included in the config if you wish to use the parameters extraction function in `mlfow_utils` to compile results.

- `defaults`: Defaults that are set for all tasks. Can be overriden under each task.

- `tasks`: List of tasks to perform. Tasks specify parameters for the decoder, datamodule to be used and training parameters.

- `n_trials`: Number of trials to be carried out per task, in the case of hyperparameter tuning.

- `save_models`: Whether to save models. Defaults to False. (Setting this to true can take up a lot of space). Models will be logged as artifacts for each run in MLFlow.

- `storage_uri`: Location to use for mlflow storage for the hyperparameter optimization (hpo) stage. During optimization, additional folders will be created in parent directory of `storage_uri`. For example, if `storage_uri` is `/opt/benchmark_experiments/hpo_results`, additional folders will include: `/opt/benchmark_experiments/hpo_repeated_exp`, `/opt/benchmark_experiments/repeated_exp_output_csv`, `/opt/benchmark_experiments/job_logs`, `/opt/benchmark_experiments/optuna_db`, etc.

- `optimization_space`: Hyperparameter space to search over. Bayesian optimization tends to work well with a small number of hyperparameters.

- `run_repetitions`: number of repetitions to be run using best setting from optimization. 

See `benchmark_v2_template.yaml` in the git repo for an example.

Besides the `--config` argument, terratorch-iterate also has two other arguments: 
* if users include `--hpo` argument, then terratorch-iterate will optimize hyperparameters. Otherwise, it will rerun best experiment 
* if users include `--repeat` argument, then terratorch-iterate will repeat the best experiment. Otherwise, terratorch-iterate will not rerun any experiment

If users want to optimize hyperparameters:
```shell
terratorch iterate --hpo --config <config-file>
```

For instance:
```shell
terratorch iterate --hpo --config configs/dofa_large_patch16_224_upernetdecoder_true_modified.yaml
```


If users want to rerun best experiment, please use the same config file. Additionally, the `parent_run_id`, which is the mlflow run id from optimization, should be added as shown below:
```shell
terratorch iterate --repeat --config <config-file> --parent_run_id <mlflow run_id from hpo>
```
For instance:
```shell
terratorch iterate --repeat --config configs/dofa_large_patch16_224_upernetdecoder_true_modified.yaml --parent_run_id 61bdee4a35a94f988ad30c46c87d4fbd
```

If users want to optimize hyperparameters then the rerun best experiment in a single command, please use both settings as shown below:
```shell
terratorch iterate --hpo --repeat --config <config-file>
```
For instance:
```shell
terratorch iterate --hpo --repeat --config configs/dofa_large_patch16_224_upernetdecoder_true_modified.yaml
```

To check the experiment results, use `mlflow ui --host $(hostname -f) --port <port> --backend-store-uri <storage_uri>` 


## Summarizing results
Summarizing results and hyperparameters of multiple experiments relies on a configuration file where the experiments and tasks are defined. This consists of:

- `list_of_experiment_names`: list of MLFLOW experiment names which had been completed.

- `task_names`: list of tasks found in each experiment in `list_of_experiment_names`.

- `run_repetitions`: number of repetitions to be expected for each experiment. This should be the same as the value used in in each experiment config file during optimization. 

- `storage_uri`: Location to use for mlflow storage for the hyperparameter optimization (hpo) stage. This should be the same value used as `storage_uri` in each experiment config file during optimization (see above).


To summarize results and hyperparameters, please run the following: 
```shell
terratorch iterate --summarize --config <summarize-config-file>
```
For instance:
```shell
terratorch iterate --hpo --repeat --config configs/summarize_results.yaml
```

The results and hyperparameters are extracted into the csv file: `summarized_results/results_and_parameters.csv`.


## Ray
You can also parallelize your runs over a ray cluster. 

Check out instructions in the [docs](./docs/ray.md)




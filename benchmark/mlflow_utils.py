import pandas as pd
import numpy as np
import glob
import mlflow
from pathlib import Path
import os
import datetime
import yaml
from mlflow.exceptions import MlflowException, MissingConfigException
import plot_tools
import seaborn as sns
import pickle
from matplotlib import pyplot as plt
import logging



def delete_nested_experiment_parent_runs(delete_runs, 
                                        experiment_id, 
                                        client, 
                                        leave_one,
                                        logger):
    experiment_ids = []
    counts = []
    runs_in_experiment = []
    for exp_parent_run_id in delete_runs:
        runs = []
        runs.append(exp_parent_run_id)
        task_parent_run_data = client.search_runs(experiment_ids=[experiment_id], 
                        filter_string=f'tags."mlflow.parentRunId" LIKE "{exp_parent_run_id}"')
        for task_parent_run in task_parent_run_data:
            task_parent_run_id = task_parent_run.info.run_id
            runs.append(task_parent_run_id)
            individual_run_data = client.search_runs(experiment_ids=[experiment_id], 
                        filter_string=f'tags."mlflow.parentRunId" LIKE "{task_parent_run_id}"')
            for individual_run in individual_run_data:
                individual_run_id = individual_run.info.run_id
                runs.append(individual_run_id)
        logger.info(f"{exp_parent_run_id}: {len(runs)}")
        experiment_ids.append(exp_parent_run_id)
        counts.append(len(runs))
        runs_in_experiment.append(runs)

    logger.info(f"counts: {counts}")
    logger.info(f"experiment_ids: {experiment_ids}")
    logger.info(f"runs_in_experiment: {len(runs_in_experiment)}")
    if leave_one and (len(counts) >0):
        index_to_keep = counts.index(max(counts))
        incomplete_run_to_finish = experiment_ids[index_to_keep]
        runs_in_experiment.pop(index_to_keep)
    else:
        incomplete_run_to_finish = None
    
    runs_to_delete = [client.delete_run(run_id) for runs in runs_in_experiment for run_id in runs]
    return incomplete_run_to_finish




def check_existing_task_parent_runs(exp_parent_run_id, 
                                    storage_uri, 
                                    experiment_name,
                                    logger):
    client = mlflow.tracking.MlflowClient(tracking_uri=storage_uri)
    experiment_info = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment_info.experiment_id

    task_parent_run_data = client.search_runs(experiment_ids=[experiment_id], 
                        filter_string=f'tags."mlflow.parentRunId" LIKE "{exp_parent_run_id}"')
    runs_to_delete = []
    complete_task_run_names = []
    all_tasks_finished = []
    #   TO DO: make sure we only have one task_parent_run for each name (needed for repeated exps)
    for task_parent_run in task_parent_run_data:
        task_run_statuses = []
        task_run_ids = []

        task_run_statuses.append(task_parent_run.info.status)
        task_run_ids.append(task_parent_run.info.run_id)
        individual_run_data = client.search_runs(experiment_ids=[experiment_id], 
                    filter_string=f'tags."mlflow.parentRunId" LIKE "{task_parent_run.info.run_id}"')
        for individual_run in individual_run_data:
            task_run_statuses.append(individual_run.info.status)
            task_run_ids.append(individual_run.info.run_id)

        task_run_statuses = list(set(task_run_statuses))
        if (len(task_run_statuses) == 1) and (task_run_statuses[0]=="FINISHED"):
            complete_task_run_names.append(task_parent_run.info.run_name)
            all_tasks_finished.append(True)
        else:
            runs_to_delete.extend(task_run_ids)
            all_tasks_finished.append(False)

    if all(all_tasks_finished) and (len(all_tasks_finished) > 0) :
        all_tasks_finished = True
    else:
        all_tasks_finished = False

    runs_to_delete = [client.delete_run(run_id) for run_id in runs_to_delete]
    complete_task_run_names = list(set(complete_task_run_names))
    return complete_task_run_names, all_tasks_finished



def check_existing_experiments(storage_uri, 
                                experiment_name, 
                                exp_parent_run_name,
                                logger):
    client = mlflow.tracking.MlflowClient(tracking_uri=storage_uri)
    experiment_info = client.get_experiment_by_name(experiment_name)

    output = {"no_existing_runs": True,
            "incomplete_run_to_finish":None,
            "finished_run":None,
            "experiment_id":None}
    if experiment_info is None:
        return output

    experiment_id = experiment_info.experiment_id
    logger.info(f"experiment_id: {experiment_id}")
    logger.info(f"experiment_name: {experiment_name}")
    output["experiment_id"] = experiment_id
    experiment_parent_run_data = client.search_runs(experiment_ids=[experiment_id], 
                            filter_string=f'tags."mlflow.runName" LIKE "{exp_parent_run_name}"')
    logger.info(f"experiment_parent_run_data: {experiment_parent_run_data}")
    if len(experiment_parent_run_data)>=1:
        logger.info("there is at least one experiment parent run")
        finished_run_id = None
        incomplete_runs = []

        #check if one of the runs is complete
        for run in experiment_parent_run_data:
            completed_task_run_names, all_tasks_finished = check_existing_task_parent_runs(run.info.run_id, 
                                                                        storage_uri, 
                                                                        experiment_name,
                                                                        logger)
            if (run.info.status == "FINISHED") and all_tasks_finished:
                finished_run_id = run.info.run_id
                logger.info(f"The following run FINISHED and will be used for repeated experiments: {finished_run_id}")
            else:
                logger.info(f"The following run {run.info.run_id} is {run.info.status}")
                incomplete_runs.append(run.info.run_id)

        
        if finished_run_id is not None:
            #delete all incomplete runs
            delete_nested_experiment_parent_runs(incomplete_runs, 
                                                    experiment_id=experiment_id, 
                                                    client=client, 
                                                    leave_one=False,
                                                    logger=logger)
            output["finished_run"] = finished_run_id
            output["no_existing_runs"] = False
        else:
            #delete all incomplete runs, leave one
            logger.info(f"incomplete_runs: {incomplete_runs}")
            output["incomplete_run_to_finish"] = delete_nested_experiment_parent_runs(incomplete_runs, 
                                                        experiment_id=experiment_id, 
                                                        client=client, 
                                                        leave_one=True,
                                                        logger=logger)
            output["no_existing_runs"] = False
    return output



def get_logger(log_level="INFO"):
    #set up logging file
    if not os.path.exists("./check_results_logs"):
        os.makedirs("check_results_logs")
    current_time = datetime.datetime.now()
    current_time = str(current_time).replace(" ", "_").replace(":", "-").replace(".", "-")
    log_file = f"check_results_logs/{current_time}"
    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler = logging.FileHandler(log_file)
    #handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.basicConfig(level=logging.CRITICAL)
    return logger

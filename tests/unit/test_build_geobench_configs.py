from pathlib import Path
import pytest
import yaml
from benchmark.config_util.build_geobench_configs import generate_iterate_config
from deepdiff import DeepDiff
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@pytest.mark.parametrize(
    "input_dir, output_dir, template, prefix, oracle_config_file",
    [
        (
            "./configs/tests/terratorch_configs/test_case_01",
            "./configs/tests/terratorch-iterate-configs/test_case_01",
            "./configs/templates/template.yaml",
            "test_config_util_",
            "./configs/tests/terratorch-iterate-configs/test_case_01/oracle/convnext_LM_iterate.yaml",
        ),
        (
            "./configs/tests/terratorch_configs/test_case_02",
            "./configs/tests/terratorch-iterate-configs/test_case_02",
            "./configs/templates/template.yaml",
            "test_config_util_",
            None,
        ),
        (
            "./configs/tests/terratorch_configs/test_case_03",
            "./configs/tests/terratorch-iterate-configs/test_case_03",
            "./configs/templates/template.yaml",
            "test_config_util_",
            None,
        ),
    ],
)
def test__generate_iterate_config(
    input_dir, output_dir, template, prefix, oracle_config_file
):
    # Get the absolute path of the current script file
    script_path = Path(__file__).resolve()

    # Get the home directory
    repo_home_dir = script_path.parent.parent
    input_dir_path: Path = repo_home_dir / input_dir
    assert input_dir_path.exists()
    assert input_dir_path.is_dir()
    output_path: Path = repo_home_dir / output_dir
    assert output_path.exists()
    assert output_path.is_dir()
    # warning! delete all files of the output dir
    for item in output_path.iterdir():
        if item.is_file():
            logging.debug(f"Cleaning up directory: {item} deleted")
            item.unlink()

    generate_iterate_config(
        input_dir=input_dir_path,
        output_dir=output_path,
        template=repo_home_dir / template,
        prefix=prefix,
    )
    generated_config_files = list(output_path.glob(f'**/{prefix}*.yaml'))
    assert len(generated_config_files) > 0

    if oracle_config_file is not None:
        oracle_path: Path = repo_home_dir / oracle_config_file
        with open(oracle_path, "r") as gt_file:
            oracle_config = yaml.safe_load(gt_file)

        for gen_config_file in generated_config_files:

            with open(gen_config_file, "r") as gen_file:
                new_config = yaml.safe_load(gen_file)

            oracle_tasks = oracle_config["tasks"]
            new_config_tasks = new_config["tasks"]
            # comparing the tasks
            for oracle_task in oracle_tasks:
                found = False
                if oracle_task.get("name") is not None:
                    del oracle_task["name"]
                for new_config_task in new_config_tasks:
                    if new_config_task.get("name") is not None:
                        del new_config_task["name"]

                    diff = DeepDiff(new_config_task, oracle_task)
                    if len(diff) == 0:
                        found = True
                    else:
                        for k in [
                            "datamodule",
                            "direction",
                            "metric",
                            "terratorch_task",
                            "type",
                        ]:
                            diff = DeepDiff(new_config_task[k], oracle_task[k])
                            assert len(diff) == 0, f"Error! {diff}"
                        found = True
                assert found, f"Error! task not found: {oracle_task}"

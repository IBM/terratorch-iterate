import itertools
from pathlib import Path
from benchmark.main import main
import pytest
import sys

CONFIG_FILES = [
    # "configs/tests/benchmark_v2_simple.yaml",
    "configs/tests/dofa_large_patch16_224_upernetdecoder_true_modified.yaml",
    "configs/tests/terratorch-iterate-configs/test_case_02/test_config_util__encoderdecoder_eo_v2_300_model_factory.yaml",
    "configs/tests/terratorch-iterate-configs/test_case_03/terratorch__encoder_decoder_timm_resnet101_model_factory.yaml",
]
# CONTINUE_EXISTING_EXPERIMENT = [True, False]
# TEST_MODELS = [True, False]
HPO = [True]
INPUT_TEST_MAIN = list(
    itertools.product(HPO, CONFIG_FILES)
)


def get_test_ids() -> list[str]:
    test_case_ids = list()
    for config, cee, tm in INPUT_TEST_MAIN:
        filename = config.split("/")[-1].replace(".yaml", "")
        tid = f"{filename}_{cee}_{tm}"
        test_case_ids.append(tid)
    return test_case_ids


@pytest.mark.parametrize(
    "config, continue_existing_experiment, test_models",
    INPUT_TEST_MAIN,
    ids=get_test_ids(),
)

# terratorch iterate --hpo --config configs/tests/benchmark_v2_simple.yaml
@pytest.mark.parametrize(
    "hpo, config",
    [
        (
            True,
            "configs/tests/terratorch-iterate-configs/test_case_02/test_config_util__encoderdecoder_eo_v2_300_model_factory.yaml",
        )
    ],
)
def test_main(
    hpo: bool,
    config: str,
    continue_existing_experiment: bool,
    test_models: bool,
):
    home_dir = Path(__file__).parent.parent.parent
    config_file: Path = home_dir / config
    assert config_file.exists()
    arguments = ["terratorch", "--config", str(config_file.resolve())]
    if hpo:
        arguments.insert(1, "--hpo")
    sys.argv = arguments
    main()

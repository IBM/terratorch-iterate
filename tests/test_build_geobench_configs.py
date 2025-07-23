from pathlib import Path
import pytest
from benchmark.config_util.build_geobench_configs import _generate_iterate_config


@pytest.mark.parametrize(
    "directory, output, template",
    [
        (
            "/Users/ltizzei/Projects/Orgs/IBM/terratorch/examples/confs/geobenchv2_detection",
            "test_examples_confs_geobenchv2_detection.yaml",
            "/Users/ltizzei/Projects/Orgs/IBM/terratorch-iterate/benchmark/config_util/geobenchv2_template.yaml",
        ),
        (
            "/Users/ltizzei/Projects/Orgs/IBM/terratorch/examples/confs/geobenchv2_detection",
            "test_v2.yaml",
            "/Users/ltizzei/Projects/Orgs/IBM/terratorch-iterate/benchmark/config_util/geobenchv2_template.yaml",
        ),
    ],
)
def test__generate_iterate_config(directory, output, template):
    directory_path = Path(directory)
    assert directory_path.exists()
    output_path = Path(__file__).parent / output
    if output_path.exists():
        print(f"Delete existing {output_path} file")
        output_path.unlink()
    _generate_iterate_config(
        directory=directory_path, output=output_path, template=template
    )
    assert output_path.exists()

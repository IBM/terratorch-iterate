import subprocess
from pathlib import Path
import time
from tests.test_benchmark import TEST_CASE_IDS


def main():

    for tc_id in TEST_CASE_IDS:
        stderr_file = f"test_benchmark-{tc_id}.err"
        stdout_file = f"test_benchmark-{tc_id}.out"

        err_file = Path.home() / stderr_file
        # delete file if it exists
        if err_file.exists():
            print(f"Delete file {err_file}")
            err_file.unlink(missing_ok=True)
            assert not err_file.exists()
        out_file = Path.home() / stdout_file

        # delete file if it exists
        if out_file.exists():
            print(f"Delete file {out_file}")
            out_file.unlink(missing_ok=True)
            assert not out_file.exists()
        jbsub = f"jbsub -e {err_file} -o {out_file} -m 40G -c 1+1 -r v100 pytest --cov-report html --cov=benchmark tests/test_benchmark.py::test_run_benchmark[{tc_id}]"
        print(f"Submitting: {jbsub}")
        cmd = jbsub.split()
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            print("Command executed successfully:")
            msg = "> is submitted to default queue"
            stdout: str = result.stdout
            assert isinstance(stdout, str)
            msg_index = stdout.find(msg)
            assert msg_index > 0
            lt_index = stdout[:msg_index].find(">")
            job_id = stdout[lt_index + 1 : msg_index]
            jbinfo = f"jbinfo {job_id}"
            cmd = jbinfo.split()
            result = subprocess.run(cmd, capture_output=True)
            stdout = result.stdout
            while result.returncode == 0 and "RUN" in stdout:
                time.sleep(30)
                result = subprocess.run(cmd, capture_output=True)

        else:
            print("Command failed with error code:", result.returncode)
            print("stderr:", result.stderr)


if __name__ == "__main__":
    main()

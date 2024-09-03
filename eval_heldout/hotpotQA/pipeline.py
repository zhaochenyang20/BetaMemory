import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path
import os, json

PREFIX = "5559"
SUFFIX = "slg_online"

MODEL_SIZES = ["8"]
TEMPERATURES = [0.0]
TASK_START_INDEXS = [0]
TASK_END_INDEXS = [50]
PROMPT_SMAPLES = ["cot"]
N_GENERATE_SAMPLES = [1]
N_EVALUATE_SAMPLES = [1]
ITERATIONS = [150]
ALGORITHMS = ["tot"]
COT_METHODS = ["knn", "random"]
COT_SIZE=range(0,11)

VARIANCES = list(
    product(
        MODEL_SIZES,
        TEMPERATURES,
        TASK_START_INDEXS,
        TASK_END_INDEXS,
        PROMPT_SMAPLES,
        N_GENERATE_SAMPLES,
        N_EVALUATE_SAMPLES,
        ITERATIONS,
        ALGORITHMS,
        COT_METHODS,
        COT_SIZE,
    )
)


HOME = Path("/data/chenyang/BetaMemory/eval_heldout/hotpotQA")
RUNNING_PATHS = [HOME / each for each in os.listdir(HOME) if "EXP" in each]


def get_running_command(
    model_size,
    temperature,
    task_start_index,
    task_end_index,
    prompt_sample,
    n_generate_sample,
    n_evaluate_sample,
    iteration,
    algorithm,
    cot_method,
    cot_size,
):
    run_name = f"{PREFIX}_EXP_{model_size}_{temperature}_{task_start_index}_{task_end_index}_{prompt_sample}_{n_generate_sample}_{n_evaluate_sample}_{iteration}_{algorithm}_{cot_method}_{cot_size}_{get_format_time()}_{SUFFIX}"
    os.system(f"mkdir {run_name}")
    log_file_path = f"{run_name}/run_log.txt"
    log_dir = f"{run_name}/log"
    command = (
        f"python run.py --model_size {model_size} --temperature {temperature} --task_start_index {task_start_index} --task_end_index {task_end_index} --cot_size {cot_size} "
        f"--prompt_sample {prompt_sample} --n_generate_sample {n_generate_sample}  --n_evaluate_sample {n_evaluate_sample}  "
        f"--iteration {iteration} --algorithm {algorithm} --cot_method {cot_method} --run_name '{run_name}' --log_dir {log_dir} "
        f"--log_file_path {log_file_path} >> {log_file_path} "
    )
    return command


def query(variance):
    (
        model_size,
        temperature,
        task_start_index,
        task_end_index,
        prompt_sample,
        n_generate_sample,
        n_evaluate_sample,
        iteration,
        algorithm,
        cot_method,
        cot_size,
    ) = variance
    run_name_prefix = f"{PREFIX}_EXP_{model_size}_{temperature}_{task_start_index}_{task_end_index}_{prompt_sample}_{n_generate_sample}_{n_evaluate_sample}_{iteration}_{algorithm}_{cot_method}_"
    running_paths = [each for each in RUNNING_PATHS if run_name_prefix in str(each)]
    if len(running_paths) == 0:
        running_command = get_running_command(
            model_size,
            temperature,
            task_start_index,
            task_end_index,
            prompt_sample,
            n_generate_sample,
            n_evaluate_sample,
            iteration,
            algorithm,
            cot_method,
            cot_size,
        )
        assert running_command is not None
        return False, [running_command]
    elif len(running_paths) == 1:
        with open(running_paths[0] / "config.json", "r", encoding="utf-8") as rf:
            info_dict = json.load(rf)
        #! 如果真的还有进程在跑，就用上面这行，如果已经被 kill 了，就用下面这行
        is_running = (
            info_dict["is_running"] if "is_running" in info_dict.keys() else False
        )
        # is_running = False
        if is_running:
            return None
        else:
            restart_command = get_running_command(
                temperature,
                task_start_index,
                task_end_index,
                prompt_sample,
                n_generate_sample,
                n_evaluate_sample,
                iteration,
                algorithm,
                cot_method,
                cot_size,
            )
            return False, [
                f"mv {str(running_paths[0])} /data/chenyang/trash",
                restart_command,
            ]

    elif len(running_paths) > 1:
        return True, running_paths


def print_and_run(command):
    print(
        f"""
=======================================================
{command}
=======================================================
"""
    )
    subprocess.run(command, shell=True)


def get_format_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H:%M")
    return str(formatted_time)


def main():
    START_COMMANDS = []
    MOVE_AND_RESTART_COMMANDS = []

    for each in VARIANCES:
        status = query(each)
        if status is not None:
            is_duplicated = status[0]
            if is_duplicated:
                print(f"Duplicate paths for {each}")
                duplicated_paths = status[1]
                for each in duplicated_paths:
                    print(f"Duplications: {each}")
            else:
                commands = status[1]
                if len(commands) == 1:
                    START_COMMANDS.append(commands[0])
                else:
                    assert len(commands) == 2
                    assert "mv" in commands[0]
                    MOVE_AND_RESTART_COMMANDS.extend(commands)

    COMMANDS = START_COMMANDS + MOVE_AND_RESTART_COMMANDS

    with ThreadPoolExecutor(max_workers=len(COMMANDS)) as executor:
        futures = [executor.submit(print_and_run, command) for command in COMMANDS]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Command generated an exception: {exc}")

    return COMMANDS


if __name__ == "__main__":
    COMMANDS = main()

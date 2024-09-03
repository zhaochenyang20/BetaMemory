import os, re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SZ = 10
PREFFIX = "555"
MIDDLE = "EXP"
SUFFIX = ""
HOME = Path("/data/chenyang/BetaMemory/eval_heldout/hotpotQA")
MAX_TRIALS = 10

running_log_dirs = [
    HOME / each
    for each in os.listdir(HOME)
    if (
        (HOME / each).is_dir()
        and (
            each.startswith(PREFFIX)
            and each.endswith(SUFFIX)
            and (MIDDLE in each)
            and (f"_{SZ}_2024" in each)
        )
    )
]


def extract_success(log_path):

    results = {}

    success_pattern = re.compile(r"^SUCCESS:\s+(\d+)$")
    end_trial_pattern = re.compile(r"End Trial #(\d+)")

    current_success = None

    with open(log_path, "r") as file:
        last_line=""
        for line in file:
            if line.strip()!="":
                last_line=line.strip()
    print("Here")
    print(eval(last_line))
    return eval(last_line)


def filter(configs):
    EXMPERIMENT_RANK = configs[0]
    TRAJACTORY_SEARCH_METHOD = configs[11]
    # if int(MEM_SIZE) == 1:
    # if int(IN_CONTEXT_TRAJACTORY_SIZE) == 1:
    if TRAJACTORY_SEARCH_METHOD == "knn":
        return True
    elif TRAJACTORY_SEARCH_METHOD == "random":
        return False
    else:
        return None


enabled = []
disabled = []
for each in [each / "run_log.txt" for each in running_log_dirs]:
    configs = (str(each).split("/")[-2]).split("_")
    status = filter(configs)
    if status is None:
        continue
    elif status:
        print(f"enabled: {each}")
        success = extract_success(str(each))
        assert len(success) == MAX_TRIALS
        enabled.append(success)
    else:
        print(f"disabled: {each}")
        success = extract_success(str(each))
        assert len(success) == MAX_TRIALS
        disabled.append(success)


import numpy as np
import matplotlib.pyplot as plt

# 假设 enabled 和 disabled 是你已经获取到的数据，格式是 list of dict
# 每个 dict 的 key 是 trial 的编号（0 ~ 9），value 是该 trial 的得分

# 初始化存储均值和标准差的列表
enabled_means = []
enabled_stds = []
disabled_means = []
disabled_stds = []

# 对每个 trial 计算均值和标准差
for trial in range(MAX_TRIALS):
    enabled_scores = [d[trial] for d in enabled]
    disabled_scores = [d[trial] for d in disabled]

    enabled_means.append(np.mean(enabled_scores))
    enabled_stds.append(np.std(enabled_scores))

    disabled_means.append(np.mean(disabled_scores))
    disabled_stds.append(np.std(disabled_scores))

# 打印结果
print("enabled Means:", enabled_means)
print("enabled STDs:", enabled_stds)
print("disabled Means:", disabled_means)
print("disabled STDs:", disabled_stds)

# 可视化均值和标准差
x = np.arange(MAX_TRIALS)

plt.figure(figsize=(12, 6))

# 绘制 enabled 的均值和标准差
plt.errorbar(x, enabled_means, yerr=enabled_stds, label="knn", fmt="-o", capsize=5)

# 绘制 disabled 的均值和标准差
plt.errorbar(x, disabled_means, yerr=disabled_stds, label="random", fmt="-o", capsize=5)

plt.xlabel("Trial")
plt.ylabel("Score")
plt.title(f"knn vs random (cot {SZ}): Mean and Std of Scores per Trial")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f"hotpot_{SZ}.png")

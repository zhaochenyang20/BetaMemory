import os
import json
import argparse
import numpy as np
from models import online_embed, build_server
import faiss
from hotpotqa import HotPotQATask
from tot import dfs_search
import logging
import random


def random_selection(lst, n=5):
    # 如果列表长度小于或等于n，直接返回整个列表
    if len(lst) <= n:
        return lst
    # 否则，从列表中随机选择n个元素
    else:
        return random.sample(lst, n)


def run(args):
    task = HotPotQATask()
    print(task)
    logs, cnt_avg, cnt_any = [], 0, 0

    # create log directories if they don't exist
    config_path = os.path.join(args.run_name, "config.json")

    with open(config_path, "w", encoding="utf-8") as wf:
        info_dict = vars(args)
        info_dict["is_running"] = True
        json.dump(info_dict, wf, indent=4)

    build_server(config_path=config_path)

    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)

    logging.basicConfig(
        filename=args.log_dir,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )

    wins = {}
    lose = {}
    succ_trjs = []
    trajectories = []
    embedding_array = np.zeros((0, 3584))
    for trial in range(10):
        print("Trial")
        print(trial)
        count = 0
        task_accs = []
        info = []
        delta_trj = []
        emb_db = faiss.IndexFlatL2(3584)
        emb_db.add(embedding_array.astype("float32"))

        for i in range(args.task_start_index, args.task_end_index):
            # solve
            if i in wins:
                continue
            prev = None
            knnret = []
            if i in lose:
                prev = lose[i]
                # doing knn here with prev lose
                # knnret=random_selection(succ_trjs,5)
                fail_vec = online_embed(str(prev))
                _, indices = emb_db.search(
                    np.array(fail_vec).reshape(1, -1).astype("float32"),
                    3,
                )
                for ind in indices[0]:
                    knnret.append(trajectories[ind])
            state, value, all_nodes, reward, em, failt, succt = dfs_search(
                args, task, i, args.iteration, knnret
            )
            if failt:
                print("Fail")
                print(i)
                lose[i] = failt[0]
            if succt:
                print("Success")
                print(i)
                wins[i] = 1
                # add succt[0] to knn pool
                succ_trjs.append(succt[0])
                delta_trj.append(succt[0])
            # log main metric
            if em is None:
                em = 0
            task_accs.append(em)
            cnt_avg = sum(task_accs) / len(task_accs)
            print(i, "len(task_accs)", len(task_accs), "cnt_avg", cnt_avg, "\n")
            # all_nodes_dict = [(node.to_dict(), value) for node, value in all_nodes]
        for trj in delta_trj:
            vec = online_embed(str(trj))
            trajectories.append(trj)
            embedding_array = np.vstack((embedding_array, np.array(vec)))
    n = args.task_end_index - args.task_start_index


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_size", type=str, default="8")
    args.add_argument("--temperature", type=float, default=1.0)
    args.add_argument("--task_start_index", type=int, default=900)
    args.add_argument("--task_end_index", type=int, default=1000)
    args.add_argument("--prompt_sample", type=str, choices=["standard", "cot"])
    args.add_argument("--n_generate_sample", type=int, default=1)
    args.add_argument("--n_evaluate_sample", type=int, default=1)
    args.add_argument("--iteration", type=int, default=50)
    args.add_argument("--algorithm", type=str, choices=["lats", "rap", "tot"])
    args.add_argument("--cot_method", type=str, choices=["knn", "random", "None"])
    args.add_argument("--run_name", type=str)
    args.add_argument("--log_file_path", type=str)
    args.add_argument("--log_dir", type=str)
    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run(args)

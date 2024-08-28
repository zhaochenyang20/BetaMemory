import os
import json
import argparse

from hotpotqa import HotPotQATask
from models import gpt_usage
from lats import lats_search
from tot import dfs_search
from rap import mcts_search
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
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    wins={}
    lose={}
    succ_trjs=[]
    for trial in range(10):
        count = 0
        task_accs = []
        info = []

        for i in range(args.task_start_index, args.task_end_index):
            # solve
            if i in wins:
                continue
            prev=None
            knnret=[]
            if i in lose:
                prev=lose[i]
                # doing knn here with prev lose
                knnret=random_selection(succ_trjs,5)
            state, value, all_nodes, reward, em ,failt,succt= dfs_search(args, task, i, args.iterations,knnret)
            if failt:
                lose[i]=failt[0]
            if succt:
                wins[i]=1
                # add succt[0] to knn pool
                succ_trjs.append(succt[0])
            # log main metric
            if em is None:
                em = 0
            task_accs.append(em)
            cnt_avg = sum(task_accs) / len(task_accs)
            print(i, 'len(task_accs)', len(task_accs), 'cnt_avg', cnt_avg, '\n')
            #all_nodes_dict = [(node.to_dict(), value) for node, value in all_nodes]
        
       
    n = args.task_end_index - args.task_start_index
    # print('usage_so_far', gpt_usage(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613'], default='gpt-3.5-turbo-0613')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])
    args.add_argument('--n_generate_sample', type=int, default=1)  
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=50)
    args.add_argument('--log', type=str)
    args.add_argument('--algorithm', type=str, choices=['lats', 'rap', 'tot'])

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)
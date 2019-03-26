from __future__ import print_function, division

import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp

import time
import numpy as np
import random
import json
from tqdm import tqdm

from utils.net_util import ScalarMeanTracker
from runners import nonadaptivea3c_val, savn_val


def main_eval(args, create_shared_model, init_agent):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = args.load_model

    processes = []

    res_queue = mp.Queue()
    if args.model == "BaseModel" or args.model == "GCN":
        args.learned_loss = False
        args.num_steps = 50
        target = nonadaptivea3c_val
    else:
        args.learned_loss = True
        args.num_steps = 6
        target = savn_val

    rank = 0
    for scene_type in args.scene_types:
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                250,
                scene_type,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()

    proc = len(args.scene_types)
    pbar = tqdm(total=250 * proc)

    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)

        tracked_means = train_scalars.pop_and_reset()

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()

    with open(args.results_json, "w") as fp:
        json.dump(tracked_means, fp, sort_keys=True, indent=4)

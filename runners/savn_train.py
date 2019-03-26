from __future__ import division

import time
import random
import setproctitle
from datasets.glove import Glove
from datasets.data import get_data

from models.model_io import ModelOptions

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    end_episode,
    transfer_gradient_to_shared,
    get_params,
    reset_player,
    SGD_step,
    compute_learned_loss,
)


def savn_train(
    rank,
    args,
    create_shared_model,
    shared_model,
    initialize_agent,
    optimizer,
    res_queue,
    end_flag,
):

    glove = Glove(args.glove_file)
    scenes, possible_targets, targets = get_data(args.scene_types, args.train_scenes)

    random.seed(args.seed + rank)
    idx = [j for j in range(len(args.scene_types))]
    random.shuffle(idx)

    setproctitle.setproctitle("Training Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    import torch

    torch.cuda.set_device(gpu_id)
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    player = initialize_agent(create_shared_model, args, rank, gpu_id=gpu_id)

    model_options = ModelOptions()

    j = 0

    while not end_flag.value:

        start_time = time.time()
        new_episode(
            args, player, scenes[idx[j]], possible_targets, targets[idx[j]], glove=glove
        )
        player.episode.exploring = True
        total_reward = 0
        player.eps_len = 0

        # theta <- shared_initialization
        params_list = [get_params(shared_model, gpu_id)]
        model_options.params = params_list[-1]
        loss_dict = {}
        reward_dict = {}
        episode_num = 0
        num_gradients = 0

        # Accumulate loss over all meta_train episodes.
        while True:
            # Run episode for k steps or until it is done or has made a mistake (if dynamic adapt is true).
            if args.verbose:
                print("New inner step")
            total_reward = run_episode(player, args, total_reward, model_options, True)

            if player.done:
                break

            if args.gradient_limit < 0 or episode_num < args.gradient_limit:

                num_gradients += 1

                # Compute the loss.
                learned_loss = compute_learned_loss(args, player, gpu_id, model_options)

                if args.verbose:
                    print("inner gradient")
                inner_gradient = torch.autograd.grad(
                    learned_loss["learned_loss"],
                    [v for _, v in params_list[episode_num].items()],
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )

                params_list.append(
                    SGD_step(params_list[episode_num], inner_gradient, args.inner_lr)
                )
                model_options.params = params_list[-1]

                # reset_player(player)
                episode_num += 1

                for k, v in learned_loss.items():
                    loss_dict["{}/{:d}".format(k, episode_num)] = v.item()

        loss = compute_loss(args, player, gpu_id, model_options)

        for k, v in loss.items():
            loss_dict[k] = v.item()
        reward_dict["total_reward"] = total_reward

        if args.verbose:
            print("meta gradient")

        # Compute the meta_gradient, i.e. differentiate w.r.t. theta.
        meta_gradient = torch.autograd.grad(
            loss["total_loss"],
            [v for _, v in params_list[0].items()],
            allow_unused=True,
        )

        end_episode(
            player,
            res_queue,
            title=args.scene_types[idx[j]],
            episode_num=0,
            total_time=time.time() - start_time,
            total_reward=total_reward,
        )

        # Copy the meta_gradient to shared_model and step.
        transfer_gradient_to_shared(meta_gradient, shared_model, gpu_id)
        optimizer.step()
        reset_player(player)

        j = (j + 1) % len(args.scene_types)

    player.exit()

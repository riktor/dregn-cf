import itertools
import time
from pathlib import Path

import dataset
import hydra
import model as cf_models
import numpy as np
import torch
from evaluator import eval_score_matrix_foldout
from logger import log_model, test_summary
from tqdm import tqdm

torch.manual_seed(2021)


def train(data_dir, cfg, logger):
    if cfg.model_type == "dregncf":
        data = dataset.PointwiseDataset(data_dir, cfg, logger)
        model = cf_models.DREGN_CF(data, cfg, logger).to(cfg.environment.device)
    else:
        raise ValueError(f"model_type: {cfg.model_type} is not valid.")

    opt = torch.optim.Adam(list(model.parameters()), lr=cfg.optimize.lr)

    data_loader = data.get_loader()
    global_s = time.time()
    training_processing_times = []
    opt.zero_grad()
    model.zero_grad()
    data_loader = data.get_loader()
    global_s = time.time()
    training_processing_times = []
    opt.zero_grad()
    model.zero_grad()
    for i, samples in enumerate(data_loader):
        epoch = i // len(data_loader)
        if epoch == cfg.optimize.train_epochs:
            break
        if i % len(data_loader) == 0:
            log_model(model, epoch - 1, cfg)

        s = time.time()
        loss, summary = model.compute_loss(epoch, i, samples)
        loss.backward()
        opt.step()
        opt.zero_grad()
        model.zero_grad()
        e = time.time()
        training_processing_times.append([s, e])
        if not cfg.log.silent:
            logger.info(summary)
    torch.cuda.empty_cache()
    global_e = time.time()
    training_processing_times = np.array(training_processing_times)
    diff = training_processing_times[1:, 1] - training_processing_times[:-1, 1]
    logger.info(f"Total time: {global_e - global_s} sec")
    logger.info(f"Number of iters: {len(training_processing_times)}")
    logger.info(f"Average time for iter: {np.average(diff)} sec")
    logger.info("Test:")
    print(test(epoch, model, data, cfg, logger))


def test(epoch, model, data, cfg, logger):
    u_batch_size = cfg.evaluation.test_batch_size

    model = model.eval()
    max_K = np.max(cfg.evaluation.topK)
    with torch.no_grad():
        test_items_dict = data.test_items_dict
        users = list(test_items_dict.keys())
        users_list = []
        rating_list = []
        total_batch = len(users) // u_batch_size + 1
        count = 0
        all_result = []

        batch_size = cfg.evaluation.test_batch_size
        for i in tqdm(range(0, len(users), batch_size), total=len(users) // batch_size):
            batch_users = users[i : i + batch_size]
            all_pos = [data.user_positive_items_dict.get(u, []) for u in batch_users]
            batch_users_gpu = (
                torch.Tensor(batch_users).long().to(cfg.environment.device)
            )

            rating = model.get_user_rating(batch_users_gpu).cpu().numpy()
            test_items = []
            for user in batch_users:
                test_items.append(data.test_items_dict[user])
            for idx, user in enumerate(batch_users):
                train_items_off = all_pos[idx]
                rating[idx][train_items_off] = -np.inf

            batch_result = eval_score_matrix_foldout(rating, test_items, max_K)
            count += len(batch_result)
            all_result.append(batch_result)

            del rating
            users_list.append(batch_users)

        all_result = np.vstack(all_result)
        return test_summary(epoch, all_result, cfg, logger)

import datetime
import glob
import logging
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from termcolor import colored


class SummaryHandler(logging.Handler):
    def __init__(self, cfg, fmt="{}: {}", delim=", ", line_fmt=None):
        super(SummaryHandler, self).__init__()
        self.cfg = cfg
        if line_fmt:
            self.line_fmt = line_fmt
        else:
            self.line_fmt = "{time} | {level: <8} | {function: ^10} | {filename: <10}:{line: >3} | {serialized}\n"
        self.fmt = fmt
        self.delim = delim

    def colorize(self, dic):
        cmap = {
            "time": "green",
            "level": lambda l: ["green", "yellow", "red"][
                ["INFO", "WARN", "ERROR"].index(l)
            ],
            "function": "white",
            "filename": "white",
            "line": "green",
            "serialized": "white",
        }
        for k in dic:
            if k not in cmap:
                dic[k] = colored(dic[k], "white")
            elif type(cmap[k]) is str:
                dic[k] = colored(dic[k], cmap[k])
            else:
                f = cmap[k]
                dic[k] = colored(dic[k], f(dic[k]))
        return dic

    def get_template(self, record):
        dic = {
            "time": datetime.datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3],
            "level": record.levelname,
            "function": record.funcName,
            "filename": record.filename,
            "line": record.lineno,
        }
        return dic

    def serialize(self, kv_list):
        tokens = []
        for k, v in kv_list:
            if type(v) is float:
                v = "{:.5}".format(v)
            tokens.append(self.fmt.format(k, v))
        line = self.delim.join(tokens)
        return line

    def emit(self, record):
        kvs = record.msg
        if type(record.msg) == dict:
            step = kvs.pop("step")
            step_name = kvs.pop("step_name")
            kv_list = [(step_name, colored(step, "green"))] + list(kvs.items())
            line = self.serialize(kv_list)
            dic = self.get_template(record)
            dic["serialized"] = line
        if type(record.msg) == str:
            dic = self.get_template(record)
            dic["serialized"] = record.msg
        dic = self.colorize(dic)
        print(self.line_fmt.format(**dic), file=sys.stdout, end="")


def getLogger(cfg, timestamp):
    logger = logging.getLogger(__name__)
    logger.propagate = False

    if not cfg.log.silent:
        logger.addHandler(SummaryHandler(cfg))

    if cfg.log.file_path:
        file_path = cfg.log.file_path
        save_path = Path(file_path) / f"log_{cfg.experiment_name}_{timestamp}.log"
        logger.addHandler(logging.FileHandler(save_path))

    return logger


def log_codes(cfg, timestamp):
    targets = glob.glob(str(Path() / "*.py"))
    enclosing_dir = Path(f"codes_{cfg.experiment_name}_{timestamp}")

    code_path = cfg.log.code_path
    save_path = Path(code_path) / f"{str(enclosing_dir)}.zip"

    with zipfile.ZipFile(save_path, "w") as zf:
        for path in targets:
            zf.write(path, enclosing_dir / path.split("/")[-1])


def log_config(cfg, timestamp):
    yaml = OmegaConf.to_yaml(cfg)
    print(yaml)
    code_path = cfg.log.code_path
    save_path = Path(code_path) / f"codes_{cfg.experiment_name}_{timestamp}.yaml"
    with open(save_path, "w") as fw:
        print(yaml, file=fw, end="")


def log_model(model, epoch, cfg):
    timestamp = time.time()
    model_path = cfg.log.model_path
    save_path = (
        Path(model_path) / f"models_{cfg.experiment_name}_{epoch}_{timestamp}.model"
    )
    torch.save(model.state_dict(), save_path)


def test_summary(epoch, all_result, cfg, logger):
    final_result = np.nanmean(all_result, axis=0)
    final_result = np.reshape(final_result, newshape=[5, np.max(cfg.evaluation.topK)])
    final_result = final_result[:, np.array(cfg.evaluation.topK) - 1]
    final_result = np.reshape(final_result, newshape=[5, len(cfg.evaluation.topK)])

    res = {"step": epoch, "step_name": "epoch"}
    for k, measure in zip(cfg.evaluation.topK, final_result[0]):
        res[f"P_{k}"] = float(measure)
    for k, measure in zip(cfg.evaluation.topK, final_result[1]):
        res[f"R_{k}"] = float(measure)
    for k, measure in zip(cfg.evaluation.topK, final_result[3]):
        res[f"nDCG_{k}"] = float(measure)

    logger.info(res)

    return res

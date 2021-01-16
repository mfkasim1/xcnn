import os
import copy
import math
import itertools
import numpy as np
from typing import Dict, Optional
import argparse
import torch
import pytorch_lightning as pl
from ray import tune
import xcdnn2
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from xcdnn2.dataset import DFTDataset
from xcdnn2.litmodule import LitDFTXC
from xcdnn2.utils import get_exp_version
from ray.tune.suggest.hyperopt import HyperOptSearch

FILEPATH = os.path.dirname(os.path.realpath(xcdnn2.__file__))

######################## hparams part ########################
def get_program_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_const", default=False, const=True,
                        help="Flag to record the progress")
    parser.add_argument("--version", type=str,
                        help="The training version, if exists, then resume the training")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="The log directory relative to this file's path")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    # hparams not used for the actual training
    # (only for different execution modes of this file)
    parser.add_argument("--cmd", action="store_const", default=False, const=True,
                        help="Run the training via command line")
    parser.add_argument("--tune", action="store_const", default=False, const=True,
                        help="Run the hyperparameters tuning")

    # training hyperparams
    parser.add_argument("--clipval", type=float, default=0,
                        help="Clip gradients with norm above this value. 0 means no clipping.")
    parser.add_argument("--max_epochs", type=int, default=1000,
                        help="Maximum number of epochs")
    parser.add_argument("--tvset", type=int, default=2,
                        help="Training/validation set")
    parser.add_argument("--exclude_types", type=str, nargs="*", default=[],
                        help="Exclude several types of dataset")
    parser.add_argument("--tiny_dset", action="store_const", default=False, const=True,
                        help="Flag to use tiny dataset for sanity check")
    return parser

def convert_to_tune_config(hparams: Dict) -> Dict:
    # set the hyperparameters to be tuned
    res = copy.deepcopy(hparams)
    split_opt = hparams["split_opt"]
    exclude_types = hparams["exclude_types"]
    res["record"] = True  # if hparams are tuned, it must be recorded

    res["nhid"] = tune.choice([16, 32, 64])
    res["ndepths"] = tune.choice([1, 2, 3, 4])
    res["ninpmode"] = tune.choice([1, 2, 3])
    res["outmultmode"] = tune.choice([1, 2])
    if (split_opt and "ie" not in exclude_types) or (not split_opt):
        res["ielr"] = tune.loguniform(1e-5, 3e-3)
    if split_opt and "ae" not in exclude_types:
        res["aelr"] = tune.loguniform(1e-5, 3e-3)
    if split_opt and "dm" not in exclude_types:
        res["dmlr"] = tune.loguniform(1e-5, 3e-3)
    if split_opt and "dens" not in exclude_types:
        res["denslr"] = tune.loguniform(1e-5, 3e-3)
    return res

######################## dataset and training part ########################
def get_datasets(hparams: Dict):
    from xcdnn2.utils import subs_present
    # load the datasets and returns the dataloader for training and validation

    # load the dataset and split into train and val
    dset = DFTDataset()
    tvset = hparams["tvset"]
    if tvset == 1:
        # train_atoms = ["H", "He", "Li", "Be", "B", "C"]
        val_atoms = ["N", "O", "F", "Ne"]
    elif tvset == 2:  # randomly selected
        # train_atoms = ["H", "Li", "B", "C", "O", "Ne"]
        val_atoms = ["He", "Be", "N", "F"]

    general_filter = lambda obj: obj["type"] not in hparams["exclude_types"]
    all_idxs = dset.get_indices(general_filter)
    val_filter = lambda obj: subs_present(val_atoms, obj["name"].split()[-1]) and general_filter(obj)
    val_idxs = dset.get_indices(val_filter)
    train_idxs = list(set(all_idxs) - set(val_idxs))
    if hparams["tiny_dset"]:
        val_idxs = val_idxs[:1]
        train_idxs = train_idxs[:1]
    # print(train_idxs, len(train_idxs))
    # print(val_idxs, len(val_idxs))
    # raise RuntimeError

    # train_idxs = range(24, 26)  # dm
    # val_idxs = range(20, 23)
    # train_idxs = range(3)  # ie
    # val_idxs = range(3, 6)
    dset_train = Subset(dset, train_idxs)
    dset_val = Subset(dset, val_idxs)
    dloader_train = DataLoader(dset_train, batch_size=None, shuffle=True)
    dloader_val = DataLoader(dset_val, batch_size=None)
    return dloader_train, dloader_val

def get_trainer(hparams: Dict):
    # setup the trainer
    trainer_kwargs = {
        "logger": False,
        "checkpoint_callback": False,
        "num_sanity_val_steps": 0,
        "gradient_clip_val": hparams["clipval"],
        "automatic_optimization": False,
        "max_epochs": hparams["max_epochs"],
    }
    logdir = os.path.join(FILEPATH, hparams["logdir"])
    version = get_exp_version(hparams["version"])
    if hparams["record"]:
        # set up the logger
        tb_logger = pl.loggers.TensorBoardLogger(logdir, version=version)
        chkpt_val = ModelCheckpoint(monitor="val_loss", save_top_k=4, save_last=True)
        trainer_kwargs["logger"] = tb_logger
        trainer_kwargs["callbacks"] = [chkpt_val]
        trainer_kwargs["checkpoint_callback"] = True
        print("Version: %s" % tb_logger.version)
    else:
        chkpt_val = None

    # resume the training from the given version
    if version is not None:
        fpath = os.path.join(logdir, "default", version, "checkpoints", "last.ckpt")
        if os.path.exists(fpath):
            print("Resuming the training from %s" % fpath)
            trainer_kwargs["resume_from_checkpoint"] = fpath

    trainer = pl.Trainer(**trainer_kwargs)
    return trainer, chkpt_val

def run_training(hparams: Dict):
    dloader_train, dloader_val = get_datasets(hparams)

    # optional step to inform the model what entries are so that it can
    # prepare specific buffer for each entries / systems
    entries = []
    for dct in itertools.chain(dloader_train, dloader_val):
        entries.append(dct)

    # create the lightning module and the datasets
    plsystem = LitDFTXC(hparams, entries)
    trainer, chkpt_val = get_trainer(hparams)
    trainer.fit(plsystem,
                train_dataloader=dloader_train,
                val_dataloaders=dloader_val)
    return chkpt_val.best_model_score.item()

############## hparams tuning part ##############
def run_training_via_cmd_line(hparams: Dict):
    # execute the training, but via command line
    # this is done because there is an unknown memory leak in the training
    # procedure

    import subprocess as sp
    cmds = ["python", os.path.join(FILEPATH, "train.py")]
    for key, val in hparams.items():
        if key == "cmd" or key == "tune":
            continue

        arg = "--" + key
        if isinstance(val, bool) and val is not None:  # a flag
            if val:
                cmds.append(arg)
        elif isinstance(val, list) or isinstance(val, tuple):
            cmds.append(arg)
            cmds.extend([str(s) for s in val])
        elif isinstance(val, str) or isinstance(val, int) or isinstance(val, float):
            cmds.append(arg)
            cmds.append(str(val))
        elif val is None:
            pass
        else:
            raise RuntimeError("Unknown type of value: %s from key '%s'" % (type(val), key))

    process = sp.run(cmds, stdout=sp.PIPE, stderr=sp.STDOUT)
    stdout = process.stdout.decode("utf-8")
    # print("stdout", stdout)

    # get the values from stdout
    val = None
    for line in stdout.split("\n"):
        if line.startswith("Output:"):
            val = float(line.split()[-1])
        elif line.startswith("Version:"):
            # set the version number so it can continue
            version = line.split()[-1]
            hparams["version"] = version

    return val

def run_training_until_complete(hparams: Dict, with_tune: bool = True):
    max_epochs = hparams["max_epochs"]
    max_epochs_1_run = 50

    # calculate how many epochs and how many repetitions needed
    n = int(math.ceil(max_epochs / max_epochs_1_run))
    epochs_1_run = int(math.ceil(max_epochs / n))

    for i in range(n):
        hparams["max_epochs"] = min(epochs_1_run * (i + 1), max_epochs)
        # print(i, hparams)
        val_loss = run_training_via_cmd_line(hparams)
        if with_tune and val_loss is not None:
            tune.report(val_loss=val_loss)
    return {"val_loss": val_loss}

def optimize_hparams(hparams: Dict):
    config = convert_to_tune_config(hparams)
    alg = HyperOptSearch(mode="min", metric="val_loss")
    alg = tune.suggest.ConcurrencyLimiter(alg, 1)
    analysis = tune.run(
        run_training_until_complete,
        config=config,
        num_samples=-1,
        search_alg=alg,
        resources_per_trial={"cpu": 8, "gpu": 0},
    )
    print("Best config:", analysis.get_best_config(metric="val_loss", mode="min"))

if __name__ == "__main__":
    # parsing the hyperparams
    parser = get_program_argparse()
    parser = LitDFTXC.get_trainer_argparse(parser)
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # set the random seed
    torch.manual_seed(args.seed)

    # putting all the hyperparameters in a dictionary
    hparams = vars(args)

    if args.cmd:
        bestval = run_training_until_complete(hparams, False)
    elif args.tune:
        bestval = optimize_hparams(hparams)
    else:
        bestval = run_training(hparams)
    print("Output:", bestval)

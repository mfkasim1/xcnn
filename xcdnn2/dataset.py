import os
from typing import Optional, Callable, Dict, List
import yaml
import torch
from torch.utils.data import Dataset
from xcdnn2.entry import Entry

filedir = os.path.dirname(os.path.realpath(__file__))

################# dataset #################

class DFTDataset(Dataset):
    def __init__(self, fpath: Optional[str] = None):
        if fpath is None:
            fpath = os.path.join(filedir, "dft_dataset.yaml")

        with open(fpath, "r") as f:
            self.obj = [Entry.create(a) for a in yaml.safe_load(f)]

    def __len__(self) -> int:
        return len(self.obj)

    def __getitem__(self, i: int) -> Entry:
        return self.obj[i]

    def get_indices(self, filtfcn: Callable[[Dict], bool]) -> List[int]:
        # return the id of the datasets that passes the filter function
        return [i for (i, obj) in enumerate(self.obj) if filtfcn(obj)]


if __name__ == "__main__":
    from xcdnn2.evaluator import XCDNNEvaluator as Evaluator
    from dqc.api.getxc import get_libxc
    import numpy as np

    dset = DFTDataset()
    weights = {
        "ie": 1.0,
        "ae": 1.0,
        "dm": 1.0,
        "dens": 1.0,
    }
    evl = Evaluator(get_libxc("lda_x"), weights)

    # run the datasets loss evaluation
    losses = {}
    for i in range(len(dset)):
        tpe = dset[i]["type"]
        if tpe not in losses:
            losses[tpe] = []
        loss = evl.calc_loss_function(dset[i])
        print(i, tpe, dset[i]["name"], loss)
        losses[tpe].append(loss.item())

    print("Type, mean loss, std loss, inv mean")
    for tpe in losses:
        print(tpe, np.mean(losses[tpe]), np.std(losses[tpe]), 1. / np.mean(losses[tpe]))

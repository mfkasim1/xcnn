import os
import yaml
import warnings
from typing import Optional, Dict, List, Union
import torch
from torch.utils.data import Dataset
from dqc.system.base_system import BaseSystem
from dqc.system.mol import Mol
from dqc.qccalc.ks import KS
from dqc.xc.base_xc import BaseXC
from dqc.api.getxc import get_libxc
from dqc.utils.datastruct import SpinParam
from xcdnn2.xcmodels import BaseNNXC

filedir = os.path.dirname(os.path.realpath(__file__))

###################### dataset ######################

class DFTDataset(Dataset):
    def __init__(self, fpath: Optional[str] = None):
        if fpath is None:
            fpath = os.path.join(filedir, "dft_dataset.yaml")

        with open(fpath, "r") as f:
            self.obj = yaml.safe_load(f)

        for i in range(len(self.obj)):
            self.obj[i]["id"] = i

    def __len__(self) -> int:
        return len(self.obj)

    def __getitem__(self, i: int) -> Dict:
        return self.obj[i]

class Evaluator(torch.nn.Module):
    def __init__(self, xc: BaseNNXC, weights: Dict[str, float]):
        super().__init__()
        self.xc = xc
        self.dm_dict: Dict[str, Union[torch.Tensor, SpinParam[torch.Tensor]]] = {}
        self.weights = weights

    ################## evaluator ##################
    def calc_loss_function(self, item: Dict) -> torch.Tensor:
        # calculate the loss function given the item
        itemtype = item["type"]

        with warnings.catch_warnings(record=True) as w:
            # evaluate the command
            val = eval(item["cmd"], self._get_glob(item["systems"]))

            # compare the calcluated value with the true value
            true_val = self._get_true_val(item["true_val"], itemtype)
            loss = self._get_loss_func(true_val, val, itemtype)

            # if there is a convergence warning, do not propagate the gradient
            if len(w) > 0:
                loss = loss * 0 + loss.detach()

        return loss

    def energy(self, system: Dict) -> torch.Tensor:
        # get the energy based on the system

        # check the cache
        system_str = str(system)
        dm0 = self.dm_dict.get(system_str, None)

        syst = self._get_system(system)
        qc = KS(syst, xc=self.xc).run(dm0=dm0)

        # save the dm cache
        self._save_dm_cache(system_str, qc.aodm())
        return qc.energy()

    def _save_dm_cache(self, s: str, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]):
        if isinstance(dm, SpinParam):
            dm.u = dm.u.detach()
            dm.d = dm.d.detach()
        else:
            dm = dm.detach()
        self.dm_dict[s] = dm

    def _get_glob(self, systems: List[Dict]):
        # return the global variables for the command evaluation
        return {
            "energy": self.energy,
            "systems": systems
        }

    def _get_loss_func(self, true_val: torch.Tensor, val: torch.Tensor, itemtype: str) -> torch.Tensor:
        w = self.weights[itemtype]
        if itemtype in ["ie", "ae"]:
            return w * torch.sum((true_val - val) ** 2)  # Hartree to kcal/mol
        else:
            raise RuntimeError("Unknown item type: %s" % itemtype)

    def _get_system(self, system: Dict) -> BaseSystem:
        # convert the system dictionary to DQC system
        # TODO: add cache
        systype = system["type"]
        if systype == "mol":
            syst = Mol(**system["kwargs"])
            return syst
        else:
            raise RuntimeError("Unknown system type: %s" % systype)

    def _get_true_val(self, trueval: Union[float, str], itemtype: str) -> torch.Tensor:
        if isinstance(trueval, float):
            return torch.tensor(trueval, dtype=torch.double)
        else:
            raise RuntimeError("Unknown item type: %s" % type(trueval))

if __name__ == "__main__":
    dset = DFTDataset()
    weights = {
        "ie": 630.0,
        "ae": 630.0,
    }
    evl = Evaluator(get_libxc("lda_x"), weights)
    for i in range(len(dset)):
        print(i, dset[i]["name"], evl.calc_loss_function(dset[i]))

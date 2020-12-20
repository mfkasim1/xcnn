import os
import yaml
import warnings
import pickle
from typing import Optional, Dict, List, Union, Tuple, Callable
import hashlib
import numpy as np
from pyscf import gto, cc, scf
import torch
import xitorch as xt
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
            # evaluate the true_val and save it to a temporary directory
            true_val = self.obj[i]["true_val"]
            if isinstance(true_val, str):
                self.obj[i]["true_val"] = _eval_true_val(self.obj[i], true_val)

    def __len__(self) -> int:
        return len(self.obj)

    def __getitem__(self, i: int) -> Dict:
        return self.obj[i]

    def get_indices(self, filtfcn: Callable[[Dict], bool]) -> List[int]:
        # return the id of the datasets that passes the filter function
        return [i for (i, obj) in enumerate(self.obj) if filtfcn(obj)]

def _eval_true_val(obji: Dict, true_val: str):
    # get the true value if true_val is a string
    # this function uses cache, so it only needs to be evaluated once

    # get the file name to store the evaluated values
    fname = str(hashlib.blake2s(str.encode(str(obji))).hexdigest()) + ".pkl"
    fdir = os.path.join(filedir, ".datasets")
    if not os.path.exists(fdir):
        os.mkdir(fdir)
    fpath = os.path.join(fdir, fname)

    # if the file exists, then load from the file, otherwise evaluate
    if os.path.exists(fpath):
        with open(fpath, "rb") as fb:
            res = pickle.load(fb)
    else:
        # evaluate the true value
        print("Evaluating the true value of '%s' and save it to %s" % \
              (obji["name"], fpath))
        res = eval(true_val, _get_true_val_glob(obji["systems"]))

        # save the result to a file
        with open(fpath, "wb") as fb:
            pickle.dump(res, fb)
    return res

def _get_true_val_glob(systems: List[Dict]) -> Dict:
    # variables in evaluating the true_val string in the dataset
    return {
        "ccsd_dm": get_pyscf_dm,
        "systems": systems,
    }

def get_pyscf_dm(system: Dict) -> torch.Tensor:
    # get the density matrix of a system with PySCF's CCSD calculation

    # run the PySCF's CCSD calculation
    mol = _get_pyscf_system(system)
    mf  = scf.UHF(mol).run()
    mcc = cc.UCCSD(mf)
    mcc.kernel()

    # obtain the total density matrix
    modm = mcc.make_rdm1()
    aodm0 = np.dot(mf.mo_coeff[0], np.dot(modm[0], mf.mo_coeff[0].T))
    aodm1 = np.dot(mf.mo_coeff[1], np.dot(modm[1], mf.mo_coeff[1].T))
    aodm = aodm0 + aodm1

    return torch.as_tensor(aodm, dtype=torch.double)

############################ evaluator ############################
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

        with warnings.catch_warnings():
            warnings.simplefilter("always", xt.MathWarning)
            warnings.simplefilter("error", xt.ConvergenceWarning)

            try:
                # evaluate the command
                val = eval(item["cmd"], self._get_glob(item["systems"]))

                # compare the calcluated value with the true value
                true_val = self._get_true_val(item["true_val"], itemtype)
                loss = self._get_loss_func(true_val, val, itemtype)

            except xt.ConvergenceWarning:
                # if there is a convergence warning, do not propagate the gradient
                loss = sum(p.sum() * 0 for p in self.xc.parameters())

        return loss

    def energy(self, system: Dict) -> torch.Tensor:
        # get the energy based on the system
        qc, dm = self._run_ks(system)
        return qc.energy()

    def dm(self, system: Dict) -> torch.Tensor:
        # get the total dm based on the system
        qc, dm = self._run_ks(system)
        if isinstance(dm, SpinParam):
            dmtot = dm.u + dm.d
        else:
            dmtot = dm
        return dmtot

    def _run_ks(self, system: Dict) -> Tuple[KS, Union[torch.Tensor, SpinParam[torch.Tensor]]]:
        # run the Kohn Sham DFT for the system

        # check the cache
        system_str = str(system)
        dm0 = self.dm_dict.get(system_str, None)

        # run ks
        syst = _get_dqc_system(system)
        qc = KS(syst, xc=self.xc).run(dm0=dm0)
        dm = qc.aodm()

        # save the cache
        self._save_dm_cache(system_str, dm)
        return qc, dm

    def _save_dm_cache(self, s: str, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]):
        if isinstance(dm, SpinParam):
            dm_copy = SpinParam(u=dm.u.detach(), d=dm.d.detach())
        else:
            dm_copy = dm.detach()
        self.dm_dict[s] = dm_copy

    def _get_glob(self, systems: List[Dict]):
        # return the global variables for the command evaluation
        return {
            "energy": self.energy,
            "dm": self.dm,
            "systems": systems
        }

    def _get_loss_func(self, true_val: torch.Tensor, val: torch.Tensor, itemtype: str) -> torch.Tensor:
        w = self.weights[itemtype]
        if itemtype in ["ie", "ae"]:
            return w * torch.sum((true_val - val) ** 2)
        elif itemtype == "dm":
            return w * torch.mean((true_val - val) ** 2)
        else:
            raise RuntimeError("Unknown item type: %s" % itemtype)

    def _get_true_val(self, true_val: Union[float, torch.Tensor, str], itemtype: str) -> torch.Tensor:
        if isinstance(true_val, float):
            return torch.tensor(true_val, dtype=torch.double)
        elif isinstance(true_val, torch.Tensor):
            return true_val
        else:
            raise RuntimeError("Unknown item type: %s" % type(true_val))

def _get_dqc_system(system: Dict) -> BaseSystem:
    # convert the system dictionary to DQC system

    systype = system["type"]
    if systype == "mol":
        return Mol(**system["kwargs"])
    else:
        raise RuntimeError("Unknown system type: %s" % systype)

def _get_pyscf_system(system: Dict):
    # convert the system dictionary PySCF system

    # TODO: add cache
    systype = system["type"]
    if systype == "mol":
        kwargs = system["kwargs"]
        return gto.M(atom=kwargs["moldesc"], basis=kwargs["basis"], spin=kwargs.get("spin", 0))
    else:
        raise RuntimeError("Unknown system type: %s" % systype)

if __name__ == "__main__":
    dset = DFTDataset()
    weights = {
        "ie": 1.0,
        "ae": 1.0,
        "dm": 1.0,
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

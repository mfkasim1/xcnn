from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Union
import warnings
import torch
import xitorch as xt
from dqc.qccalc.base_qccalc import BaseQCCalc
from dqc.system.mol import Mol
from dqc.qccalc.ks import KS
from dqc.utils.datastruct import SpinParam
from xcdnn2.entry import Entry, System
from xcdnn2.xcmodels import BaseNNXC

class BaseEvaluator(torch.nn.Module):
    """
    Object containing trainable parameters and the interface to the NN models.
    """
    @abstractmethod
    def calc_loss_function(self, entry: Entry) -> torch.Tensor:
        """
        Calculate the weighted loss function using the parameters.
        This is like the forward of standard torch nn Module.
        """
        pass

    @abstractmethod
    def run(self, system: System) -> BaseQCCalc:
        """
        Run the Quantum Chemistry calculation of the given system and return
        the post-run QCCalc object
        """
        pass

class XCDNNEvaluator(BaseEvaluator):
    """
    Kohn-Sham model where the XC functional is replaced by a neural network
    """
    def __init__(self, xc: BaseNNXC, weights: Dict[str, float]):
        super().__init__()
        self.xc = xc
        self.weights = weights

    def calc_loss_function(self, entry_raw: Union[Entry, Dict]) -> torch.Tensor:
        # calculate the loss function given the entry

        # get the entry object
        if isinstance(entry_raw, dict):
            entry = Entry.create(entry_raw)
        else:
            entry = entry_raw

        w = self.weights[entry["type"]]

        with warnings.catch_warnings():
            warnings.simplefilter("always", xt.MathWarning)
            warnings.simplefilter("error", xt.ConvergenceWarning)

            try:
                # evaluate the command
                qcs = [self.run(syst) for syst in entry.get_systems()]
                val = entry.get_val(qcs)

                # compare the calculated value with the true value
                true_val = entry.get_true_val()
                loss = w * entry.get_loss(val, true_val)

            except xt.ConvergenceWarning:
                # if there is a convergence warning, do not propagate the gradient
                loss = sum(p.sum() * 0 for p in self.xc.parameters())

        return loss

    def run(self, system: System) -> BaseQCCalc:
        # run the Kohn Sham DFT for the system

        # check the cache
        system_str = str(system)
        dm0 = system.get_cache("dm0")

        # run ks
        syst = system.get_dqc_system()
        qc = KS(syst, xc=self.xc).run(dm0=dm0, bck_options={"max_niter": 50})
        dm = qc.aodm()

        # save the cache
        if isinstance(dm, SpinParam):
            dm_cache = SpinParam(u=dm.u.detach(), d=dm.d.detach())
        else:
            dm_cache = dm.detach()
        system.set_cache("dm0", dm_cache)
        return qc

from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Union
import warnings
import torch
import xitorch as xt
from dqc.system.mol import Mol
from dqc.qccalc.ks import KS
from dqc.utils.datastruct import SpinParam
from pyscf import dft
from xcdnn2.entry import Entry, System
from xcdnn2.kscalc import BaseKSCalc, DQCKSCalc, PySCFKSCalc
from xcdnn2.xcmodels import BaseNNXC

class BaseEvaluator(torch.nn.Module):
    """
    Object containing trainable parameters and the interface to the NN models.
    """
    @abstractmethod
    def calc_loss_function(self, entry_raw: Union[Entry, Dict]) -> torch.Tensor:
        """
        Calculate the weighted loss function using the parameters.
        This is like the forward of standard torch nn Module.
        """
        pass

    @abstractmethod
    def calc_deviation(self, entry_raw: Union[Entry, Dict]) -> torch.Tensor:
        """
        Calculates and returns deviation in a readable unit and interpretation.
        """
        pass

    @abstractmethod
    def run(self, system: System) -> BaseKSCalc:
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
        # calculate the loss function of the entry
        fcn = lambda entry, val, true_val: self.weights[entry["type"]] * entry.get_loss(val, true_val)
        return self._calc_loss(entry_raw, fcn)

    def calc_deviation(self, entry_raw: Union[Entry, Dict]) -> torch.Tensor:
        # calculate the deviation from the true value in an interpretable format
        fcn = lambda entry, val, true_val: entry.get_deviation(val, true_val)
        return self._calc_loss(entry_raw, fcn)

    def _calc_loss(self, entry_raw: Union[Entry, Dict], fcn: Callable) -> torch.Tensor:
        # calculate the loss function given the entry and function to calculate the loss

        # get the entry object
        entry = Entry.create(entry_raw)

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always", xt.MathWarning)
            warnings.simplefilter("always", xt.ConvergenceWarning)

            # evaluate the command
            qcs = [self.run(syst) for syst in entry.get_systems()]
            val = entry.get_val(qcs)

            # compare the calculated value with the true value
            true_val = entry.get_true_val()
            loss = fcn(entry, val, true_val)

        # if there is a warning, show all of them
        if len(ws) > 0:
            convergence_warning = False
            for w in ws:
                warnings.warn(w.message, category=w.category)
                if issubclass(w.category, xt.ConvergenceWarning):
                    convergence_warning = True
            if convergence_warning:
                # if there is a convergence warning, do not propagate the gradient,
                # but preserve the value
                loss = sum(p.sum() * 0 for p in self.xc.parameters()) + loss.detach()
                print("Evaluation of '%s' is not converged" % entry["name"])

        return loss

    def run(self, system: System) -> BaseKSCalc:
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
        return DQCKSCalc(qc)

class PySCFEvaluator(BaseEvaluator):
    """
    KS model using PySCF.
    """
    def __init__(self, xc: str, weights: Dict[str, float]):
        super().__init__()
        self.xc = xc
        self.weights = weights

    def calc_loss_function(self, entry_raw: Union[Entry, Dict]) -> torch.Tensor:
        # calculate the loss function of the entry
        fcn = lambda entry, val, true_val: self.weights[entry["type"]] * entry.get_loss(val, true_val)
        return self._calc_loss(entry_raw, fcn)

    def calc_deviation(self, entry_raw: Union[Entry, Dict]) -> torch.Tensor:
        # calculate the deviation from the true value in an interpretable format
        fcn = lambda entry, val, true_val: entry.get_deviation(val, true_val)
        return self._calc_loss(entry_raw, fcn)

    def _calc_loss(self, entry_raw: Union[Entry, Dict], fcn: Callable) -> torch.Tensor:
        # calculate the loss function given the entry and function to calculate the loss

        # get the entry object
        entry = Entry.create(entry_raw)

        # evaluate the command
        qcs = [self.run(syst) for syst in entry.get_systems()]
        val = entry.get_val(qcs)

        # compare the calculated value with the true value
        true_val = entry.get_true_val()
        loss = fcn(entry, val, true_val)

        return loss

    def run(self, system: System) -> BaseKSCalc:
        # run the Kohn Sham DFT for the system

        # run ks in pyscf
        syst = system.get_pyscf_system()
        if syst.spin == 0:
            qc = dft.RKS(syst)
        else:
            qc = dft.UKS(syst)
        qc.xc = self.xc
        qc.kernel()

        return PySCFKSCalc(qc, syst)

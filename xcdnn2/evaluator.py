from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Union, List, Optional
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
from xcdnn2.utils import hashstr

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
    def __init__(self, xc: BaseNNXC, weights: Dict[str, float],
                 always_attach: bool = False,
                 entries: List[Dict] = []):
        super().__init__()
        self.xc = xc
        self.weights = weights
        self.always_attach = always_attach  # always attach even if the iteration does not converge

        # register system-specific buffer
        self._init_dm_buffer(entries)

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
                if not self.always_attach:
                    loss = sum(p.sum() * 0 for p in self.xc.parameters()) + loss.detach()
                print("Evaluation of '%s' is not converged" % entry["name"])

        return loss

    def run(self, system: System) -> BaseKSCalc:
        # run the Kohn Sham DFT for the system

        # check the buffer for the initial density matrix
        dm0, buffer_name = self._get_dm0_buffer(system)

        # run ks
        syst = system.get_dqc_system()
        qc = KS(syst, xc=self.xc).run(dm0=dm0, bck_options={"max_niter": 50})
        dm = qc.aodm()

        # save the cache
        if isinstance(dm, SpinParam):
            dm_cache = torch.cat((dm.u.detach().unsqueeze(0), dm.d.detach().unsqueeze(0)), dim=0)
            # dm_cache = SpinParam(u=dm.u.detach(), d=dm.d.detach())
        else:
            dm_cache = dm.detach()

        # save the buffer
        # Do not saving buffer which didn't exist before so that when the model
        # is loaded from a checkpoint, there is no new buffer (which will raise
        # error of unknown key)
        if buffer_name is not None:
            self.register_buffer(buffer_name, dm_cache)
        # system.set_cache("dm0", dm_cache)
        return DQCKSCalc(qc)

    def _dm0_buffer_name(self, obj) -> str:
        # returns the buffer name
        return "dm0_" + hashstr(str(obj))

    def _init_dm_buffer(self, entries: List[Dict]) -> None:
        # initialize the dm0 cache for each system in the entries as buffer
        for entry_dct in entries:
            entry = Entry.create(entry_dct)
            systems = entry.get_systems()
            for syst in systems:
                buffer_name = self._dm0_buffer_name(syst)
                dqc_syst = syst.get_dqc_system()
                dqc_hamilt = dqc_syst.get_hamiltonian()
                dqc_hamilt.build()
                nao = dqc_hamilt.nao
                if dqc_syst.spin != 0:
                    shape = (2, nao, nao)
                else:
                    shape = (nao, nao)
                val = torch.zeros(shape, dtype=torch.double)
                self.register_buffer(buffer_name, val)

    def _get_dm0_buffer(self, system: System) -> \
        Tuple[Union[None, torch.Tensor, SpinParam[torch.Tensor]], Optional[str]]:
        # get the dm0 cache from the buffer

        # Returns a tuple of the dm0 which is a tensor if it has been written or
        # None if no dm0 has been stored before
        # and the buffer name if the buffer is created during initialization or
        # None otherwise
        buffer_name = self._dm0_buffer_name(system)
        dm0: Optional[torch.Tensor] = getattr(self, buffer_name, None)

        buffer_exists = dm0 is not None
        buffer_written = buffer_exists and torch.any(dm0 != 0.0)
        if not buffer_written:
            dm0_res: Union[None, torch.Tensor, SpinParam[torch.Tensor]] = None
        elif system.get_dqc_system().spin != 0:
            dm0_res = SpinParam(u=dm0[0].detach(), d=dm0[1].detach())
        else:
            dm0_res = dm0

        return dm0_res, (buffer_name if buffer_exists else None)

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

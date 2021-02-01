from abc import abstractmethod
import torch
from pyscf.dft import numint
from dqc.utils.datastruct import SpinParam
from dqc.qccalc.base_qccalc import BaseQCCalc

class BaseKSCalc(object):
    """
    Interface to various quantum chemistry calculation from different softwares.
    """
    @abstractmethod
    def energy(self) -> torch.Tensor:
        # returns the energy of the Kohn-Sham calculation
        pass

    @abstractmethod
    def aodmtot(self) -> torch.Tensor:
        # returns the total density matrix in atomic orbital bases
        pass

    @abstractmethod
    def dens(self, rgrid: torch.Tensor) -> torch.Tensor:
        # returns the density profile in the given rgrid (npoints, 3)
        pass

class DQCKSCalc(BaseKSCalc):
    """
    Interface to DQC's KS calculation.
    """
    def __init__(self, qc: BaseQCCalc):
        self.qc = qc

    def energy(self) -> torch.Tensor:
        # returns the total energy
        return self.qc.energy()

    def aodmtot(self) -> torch.Tensor:
        # returns the total density matrix
        dm = self.qc.aodm()
        if isinstance(dm, SpinParam):
            dmtot = dm.u + dm.d
        else:
            dmtot = dm
        return dmtot

    def dens(self, rgrid: torch.Tensor) -> torch.Tensor:
        # returns the total density profile in the given grid
        dmtot = self.aodmtot()
        return self.qc.get_system().get_hamiltonian().aodm2dens(dmtot, rgrid)

class PySCFKSCalc(BaseKSCalc):
    """
    Interface to PySCF's KS calculation
    """
    def __init__(self, qc, mol, with_t_corr=False):
        self.qc = qc
        self.mol = mol
        self.polarized = mol.spin != 0
        self.with_t_corr = with_t_corr  # with triplet correction, only for CCSD object

    def energy(self) -> torch.Tensor:
        e_tot = self.qc.e_tot
        if self.with_t_corr:
            e_corr = self.qc.ccsd_t()
            e_tot = e_tot + e_corr
        return torch.as_tensor(e_tot)

    def aodmtot(self) -> torch.Tensor:
        dm = self.qc.make_rdm1()
        if self.polarized:
            dmtot = dm[0] + dm[1]
        else:
            dmtot = dm
        return torch.as_tensor(dmtot)

    def dens(self, rgrid: torch.Tensor) -> torch.Tensor:
        dmtot = self.aodmtot().detach().numpy()
        ao = numint.eval_ao(self.mol, rgrid.detach())
        dens = numint.eval_rho(self.mol, ao, dmtot)  # (*BG, ngrid)
        return torch.as_tensor(dens)

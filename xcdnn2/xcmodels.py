from abc import abstractproperty, abstractmethod
import torch
from typing import Union, Iterator, List
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import ValGrad, SpinParam
from dqc.utils.safeops import safenorm
from dqc.api.getxc import get_libxc

class BaseNNXC(BaseXC, torch.nn.Module):
    @abstractproperty
    def family(self) -> int:
        pass

    @abstractmethod
    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        pass

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        # torch.nn.module prefix has no ending dot, while xt prefix has
        nnprefix = prefix if prefix == "" else prefix[:-1]
        return [name for (name, param) in self.named_parameters(prefix=nnprefix)]

class NNLDA(BaseNNXC):
    # neural network xc functional of LDA (only receives the density as input)

    def __init__(self, nnmodel: torch.nn.Module):
        # nnmodel should receives input with shape (..., 2)
        # where the last dimension is for:
        # (0) total density: (n_up + n_dn), and
        # (1) spin density: (n_up - n_dn) / (n_up + n_dn)
        # the output of the model must have shape of (..., 1)
        # it represents the energy density per density per volume
        super().__init__()
        self.nnmodel = nnmodel

    @property
    def family(self) -> int:
        return 1

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        # densinfo.value: (*BD, nr)
        # collect the total density (n) and the spin density (xi)
        if isinstance(densinfo, ValGrad):  # unpolarized case
            n = densinfo.value.unsqueeze(-1)  # (*BD, nr, 1)
            xi = torch.zeros_like(n)
        else:  # polarized case
            nu = densinfo.u.value.unsqueeze(-1)
            nd = densinfo.d.value.unsqueeze(-1)
            n = nu + nd  # (*BD, nr, 1)
            xi = (nu - nd) / (n + 1e-18)  # avoiding nan

        x = torch.cat((n, xi), dim=-1)  # (*BD, nr, 2)
        res = self.nnmodel(x) * n  # (*BD, nr)
        res = res.squeeze(-1)
        return res

class NNGGA(BaseNNXC):
    # neural network xc functional of GGA (receives the density and grad as inputs)

    def __init__(self, nnmodel: torch.nn.Module):
        # nnmodel should receives input with shape (..., 3)
        # where the last dimension is for:
        # (0) total density (n): (n_up + n_dn), and
        # (1) spin density (xi): (n_up - n_dn) / (n_up + n_dn)
        # (2) normalized gradients (s): |del(n)| / [2(3*pi^2)^(1/3) * n^(4/3)]
        # the output of the model must have shape of (..., 1)
        # it represents the energy density per density per volume
        super().__init__()
        self.nnmodel = nnmodel

    @property
    def family(self) -> int:
        return 2

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        # densinfo.value: (*BD, nr)
        # densinfo.grad : (*BD, nr, 3)

        # collect the total density (n), spin density (xi), and normalized gradients (s)
        a = 6.187335452560271  # 2 * (3 * np.pi ** 2) ** (1.0 / 3)
        if isinstance(densinfo, ValGrad):  # unpolarized case
            assert densinfo.grad is not None
            n = densinfo.value.unsqueeze(-1)  # (*BD, nr, 1)
            xi = torch.zeros_like(n)
            n_offset = n + 1e-18  # avoiding nan
            s = safenorm(densinfo.grad, dim=-1).unsqueeze(-1) / (a * n_offset ** (4.0 / 3))
        else:  # polarized case
            assert densinfo.u.grad is not None
            assert densinfo.d.grad is not None
            nu = densinfo.u.value.unsqueeze(-1)
            nd = densinfo.d.value.unsqueeze(-1)
            n = nu + nd  # (*BD, nr, 1)
            n_offset = n + 1e-18  # avoiding nan
            xi = (nu - nd) / n_offset
            s = safenorm(densinfo.u.grad + densinfo.d.grad, dim=-1).unsqueeze(-1) / (a * n_offset ** (4.0 / 3))

        x = torch.cat((n, xi, s), dim=-1)  # (*BD, nr, 3)
        res = self.nnmodel(x) * n  # (*BD, nr)
        res = res.squeeze(-1)
        return res

class HybridXC(BaseNNXC):
    def __init__(self, xcstr: str, nnmodel: torch.nn.Module,
                 aweight0: float = 0.0,  # weight of the neural network
                 bweight0: float = 1.0,  # weight of the default xc
                 dtype: torch.dtype = torch.double,
                 device: torch.device = torch.device("cpu")):
        # hybrid libxc and neural network xc where it starts as libxc and then
        # trains the weights of libxc and nn xc

        super().__init__()
        self.xc = get_libxc(xcstr)
        if self.xc.family == 1:
            self.nnxc = NNLDA(nnmodel)
        elif self.xc.family == 2:
            self.nnxc = NNGGA(nnmodel)
        elif self.xc.family == 3:
            self.nnxc = NNMGGA(nnmodel)

        self.aweight = torch.nn.Parameter(torch.tensor(aweight0, dtype=dtype, device=device, requires_grad=True))
        self.bweight = torch.nn.Parameter(torch.tensor(bweight0, dtype=dtype, device=device, requires_grad=True))
        self.weight_activation = torch.nn.Identity()

    @property
    def family(self) -> int:
        return self.xc.family

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        nnlda_ene = self.nnxc.get_edensityxc(densinfo)
        lda_ene = self.xc.get_edensityxc(densinfo)
        aweight = self.weight_activation(self.aweight)
        bweight = self.weight_activation(self.bweight)
        return nnlda_ene * aweight + lda_ene * bweight

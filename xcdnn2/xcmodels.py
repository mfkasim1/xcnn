from abc import abstractproperty, abstractmethod
import torch
import numpy as np
from typing import Union, Iterator, List
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import ValGrad, SpinParam
from dqc.utils.safeops import safenorm, safepow
from dqc.api.getxc import get_xc

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

    def __init__(self, nnmodel: torch.nn.Module, ninpmode: int = 1, outmultmode: int = 1):
        # nnmodel should receives input with shape (..., 2)
        # where the last dimension is for:
        # (0) total density: (n_up + n_dn), and
        # (1) spin density: (n_up - n_dn) / (n_up + n_dn)
        # the output of the model must have shape of (..., 1)
        # it represents the energy density per density per volume
        super().__init__()
        self.nnmodel = nnmodel
        self.ninpmode = ninpmode
        self.outmultmode = outmultmode

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

        # decide how to transform the density to be the input of nn
        ninp = get_n_input(n, self.ninpmode)

        # get the neural network output
        x = torch.cat((ninp, xi), dim=-1)  # (*BD, nr, 2)
        nnout = self.nnmodel(x)  # (*BD, nr, 1)
        res = get_out_from_nnout(nnout, n, self.outmultmode)  # (*BD, nr, 1)

        # # decide how to calculate the Exc from the NN output
        # # not removed for information of the past versions
        # if self.nnxcmode == 1:
        #     x = torch.cat((n, xi), dim=-1)  # (*BD, nr, 2)
        #     res = self.nnmodel(x) * n
        # elif self.nnxcmode == 2:
        #     n_cbrt = safepow(n, 1.0 / 3)
        #     exunif = b * n_cbrt
        #     x = torch.cat((n_cbrt, xi), dim=-1)  # (*BD, nr, 2)
        #     res = self.nnmodel(x) * n * exunif  # (*BD, nr)
        # elif self.nnxcmode == 3:
        #     n_cbrt = safepow(n, 1.0 / 3)
        #     x = torch.cat((n_cbrt, xi), dim=-1)  # (*BD, nr, 2)
        #     res = self.nnmodel(x) * n  # (*BD, nr)
        # else:
        #     raise RuntimeError("Unknown nnxcmode: %d" % self.nnxcmode)
        res = res.squeeze(-1)
        return res

class NNGGA(BaseNNXC):
    # neural network xc functional of GGA (receives the density and grad as inputs)

    def __init__(self, nnmodel: torch.nn.Module, ninpmode: int = 1, outmultmode: int = 1):
        # nnmodel should receives input with shape (..., 3)
        # where the last dimension is for:
        # (0) total density (n): (n_up + n_dn), and
        # (1) spin density (xi): (n_up - n_dn) / (n_up + n_dn)
        # (2) normalized gradients (s): |del(n)| / [2(3*pi^2)^(1/3) * n^(4/3)]
        # the output of the model must have shape of (..., 1)
        # it represents the energy density per density per volume
        super().__init__()
        self.nnmodel = nnmodel
        self.ninpmode = ninpmode
        self.outmultmode = outmultmode

    @property
    def family(self) -> int:
        return 2

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        # densinfo.value: (*BD, nr)
        # densinfo.grad : (*BD, nr, 3)

        # collect the total density (n), spin density (xi), and normalized gradients (s)
        a = 6.187335452560271  # 2 * (3 * np.pi ** 2) ** (1.0 / 3)
        b = -0.7385587663820223  # -0.75 / np.pi * (3*np.pi**2)**(1./3) for exunif
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

        # decide how to transform the density to be the input of nn
        ninp = get_n_input(n, self.ninpmode)

        # get the neural network output
        x = torch.cat((ninp, xi, s), dim=-1)  # (*BD, nr, 3)
        nnout = self.nnmodel(x)  # (*BD, nr, 1)
        res = get_out_from_nnout(nnout, n, self.outmultmode)  # (*BD, nr, 1)

        # # decide how to calculate the Exc from the NN output
        # if self.nnxcmode == 1:
        #     x = torch.cat((n, xi, s), dim=-1)  # (*BD, nr, 3)
        #     res = self.nnmodel(x) * n  # (*BD, nr)
        # elif self.nnxcmode == 2:
        #     n_cbrt = safepow(n, 1.0 / 3)
        #     exunif = b * n_cbrt
        #     x = torch.cat((n_cbrt, xi, s), dim=-1)  # (*BD, nr, 3)
        #     res = self.nnmodel(x) * n * exunif  # (*BD, nr)
        # elif self.nnxcmode == 3:
        #     n_cbrt = safepow(n, 1.0 / 3)
        #     x = torch.cat((n_cbrt, xi, s), dim=-1)  # (*BD, nr, 3)
        #     res = self.nnmodel(x) * n  # (*BD, nr)
        # else:
        #     raise RuntimeError("Unknown nnxcmode: %d" % self.nnxcmode)
        res = res.squeeze(-1)
        return res

class HybridXC(BaseNNXC):
    def __init__(self, xcstr: str, nnmodel: torch.nn.Module, *,
                 ninpmode: int = 1,  # mode to decide how to transform the density to nn input
                 outmultmode: int = 1,  # mode of calculating Eks from output of nn
                 aweight0: float = 0.0,  # weight of the neural network
                 bweight0: float = 1.0,  # weight of the default xc
                 dtype: torch.dtype = torch.double,
                 device: torch.device = torch.device("cpu")):
        # hybrid libxc and neural network xc where it starts as libxc and then
        # trains the weights of libxc and nn xc

        super().__init__()
        self.xc = get_xc(xcstr)
        if self.xc.family == 1:
            self.nnxc = NNLDA(nnmodel, ninpmode=ninpmode, outmultmode=outmultmode)
        elif self.xc.family == 2:
            self.nnxc = NNGGA(nnmodel, ninpmode=ninpmode, outmultmode=outmultmode)
        elif self.xc.family == 3:
            self.nnxc = NNMGGA(nnmodel, ninpmode=ninpmode, outmultmode=outmultmode)

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

##################### supporting functions #####################
def get_n_input(n: torch.Tensor, ninpmode: int) -> torch.Tensor:
    # transform the density to the input of the neural network
    if ninpmode == 1:
        return n
    elif ninpmode == 2:
        return safepow(n, 1.0 / 3)
    elif ninpmode == 3:
        return torch.log1p(n)
    else:
        raise RuntimeError("Unknown ninpmode: %d" % ninpmode)

def get_out_from_nnout(nnout: torch.Tensor, n: torch.Tensor, outmultmode: int) -> torch.Tensor:
    # calculate the energy density per volume given the density and output
    # of the neural network
    if outmultmode == 1:
        return nnout * n
    elif outmultmode == 2:
        b = -0.7385587663820223  # -0.75 / np.pi * (3*np.pi**2)**(1./3) for exunif
        exunif = b * safepow(n, 1.0 / 3)
        return nnout * n * exunif
    else:
        raise RuntimeError("Unknown outmultmode: %d" % self.outmultmode)

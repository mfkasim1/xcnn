from abc import abstractproperty, abstractmethod
import numpy as np
import math
from typing import Union
import torch
import torch.nn as nn
import dqc.xc
import dqc.utils
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import ValGrad, SpinParam


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


class pcNN_MGGA(BaseNNXC):
    # neural network xc functional of pcNN (metaGGA)

    def __init__(self, seed=0, dtype=torch.float64):
        super().__init__()
        ### parameters ###
        self.hidden = 100  # layer size
        self.dtype = dtype
        self.tanhsigma = 1.0  # decaying speed of arctan
        self.DELTA_X = 1.0  # delta to suppress higher order effect in polynomial
        self.DELTA_C = 1.0
        #################
        torch.set_default_dtype(dtype)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.fc1 = nn.Linear(2, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.hidden)
        self.fc4 = nn.Linear(self.hidden, 1)

        self.fc1c = nn.Linear(2, self.hidden)
        self.fc2c = nn.Linear(self.hidden, self.hidden)
        self.fc3c = nn.Linear(self.hidden*2, self.hidden)
        self.fc4c = nn.Linear(self.hidden, 1)

    @property
    def family(self):
        # 1 for LDA, 2 for GGA, 4 for MGGA
        return 4

    def load(self, fname):
        # debug = 1.0
        self.w1 = nn.Parameter(torch.tensor(
            np.load(fname+"/w1.npy"), dtype=self.dtype))
        self.w2 = nn.Parameter(torch.tensor(
            np.load(fname+"/w2.npy"), dtype=self.dtype))
        self.w3 = nn.Parameter(torch.tensor(
            np.load(fname+"/w3.npy"), dtype=self.dtype))
        self.w4 = nn.Parameter(torch.tensor(
            np.load(fname+"/w4.npy"), dtype=self.dtype))
        self.w1c = nn.Parameter(torch.tensor(
            np.load(fname+"/w1c.npy"), dtype=self.dtype))
        self.w2c = nn.Parameter(torch.tensor(
            np.load(fname+"/w2c.npy"), dtype=self.dtype))
        self.w3c = nn.Parameter(torch.tensor(
            np.load(fname+"/w3c.npy"), dtype=self.dtype))
        self.w4c = nn.Parameter(torch.tensor(
            np.load(fname+"/w4c.npy"), dtype=self.dtype))
        self.b1 = nn.Parameter(torch.tensor(
            np.load(fname+"/b1.npy"), dtype=self.dtype))
        self.b2 = nn.Parameter(torch.tensor(
            np.load(fname+"/b2.npy"), dtype=self.dtype))
        self.b3 = nn.Parameter(torch.tensor(
            np.load(fname+"/b3.npy"), dtype=self.dtype))
        self.b4 = nn.Parameter(torch.tensor(
            np.load(fname+"/b4.npy"), dtype=self.dtype))
        self.b1c = nn.Parameter(torch.tensor(
            np.load(fname+"/b1c.npy"), dtype=self.dtype))
        self.b2c = nn.Parameter(torch.tensor(
            np.load(fname+"/b2c.npy"), dtype=self.dtype))
        self.b3c = nn.Parameter(torch.tensor(
            np.load(fname+"/b3c.npy"), dtype=self.dtype))
        self.b4c = nn.Parameter(torch.tensor(
            np.load(fname+"/b4c.npy"), dtype=self.dtype))

        self.fc1.weight = self.w1
        self.fc2.weight = self.w2
        self.fc3.weight = self.w3
        self.fc4.weight = self.w4
        self.fc1c.weight = self.w1c
        self.fc2c.weight = self.w2c
        self.fc3c.weight = self.w3c
        self.fc4c.weight = self.w4c
        self.fc1.bias = self.b1
        self.fc2.bias = self.b2
        self.fc3.bias = self.b3
        self.fc4.bias = self.b4
        self.fc1c.bias = self.b1c
        self.fc2c.bias = self.b2c
        self.fc3c.bias = self.b3c
        self.fc4c.bias = self.b4c

    def makeg(self, x):
        unif = (x[:, 1]+x[:, 0]+1e-7)**(1.0/3.0)

        t0 = unif
        div = 1.0/(x[:, 1]+x[:, 0]+1e-7)
        t1 = ((1+(x[:, 0]-x[:, 1])*div)**(4.0/3.0) +
              (1-(x[:, 0]-x[:, 1])*div)**(4.0/3))*0.5
        t2 = ((x[:, 2]+x[:, 4]+2.0*x[:, 3]+0.1**(56/3))**0.5)/unif**4
        tauunif = 2.871234000188191*unif**5  # 0.3*(3*PI**2)**(2.0/3.0)
        t3 = (x[:, 5]+x[:, 6])/tauunif-1.0

        t = torch.tanh(torch.stack((t0, t1, t2, t3), dim=-1)/self.tanhsigma)
        return t

    def shifted_softplus0(self, x):
        l2 = 0.6931471805599453
        tmp = torch.exp(2.0*l2*(x))
        f = 1.0/l2*torch.log(1+tmp)
        return f

    def shifted_softplus1(self, x):
        l2 = 0.6931471805599453
        tmp = torch.exp(2.0*l2*(x-1.0))
        f = 1.0/l2*torch.log(1+tmp)
        return f

    def forward(self, t):
        g1 = self.shifted_softplus0(self.fc1(t))
        g2 = self.shifted_softplus0(self.fc2(g1))
        g3 = self.shifted_softplus0(self.fc3(g2))
        g4 = self.fc4(g3)
        return g4
        # return 1.0

    def forward_c(self, t):
        t1 = t[:, 0:2]  # rho, zeta
        t2 = t[:, 2:4]  # s, tau

        g1 = self.shifted_softplus0(self.fc1c(t1))
        g2 = self.shifted_softplus0(self.fc2c(g1))

        g1_x = self.shifted_softplus0(self.fc1(t2))
        g2_x = self.shifted_softplus0(self.fc2(g1_x))

        g2_c = torch.cat((g2, g2_x), dim=1)

        g3 = self.shifted_softplus0(self.fc3c(g2_c))
        g4 = self.fc4c(g3)
        return g4

    def gen_conds(self, N, g):
        s, tau = g[:, 0], g[:, 1]

        zeros = torch.zeros(N, dtype=self.dtype)
        ones = torch.ones(N, dtype=self.dtype)

        g0_1 = torch.stack((zeros, zeros), -1)
        g0_2 = torch.stack((ones, tau), -1)
        g0s = torch.stack((g0_1, g0_2), 1)  # g0s: (grids, N_conditiolns, 2)
        self.ncon_x = 2  # g0s.shape[0]
        c_s_inf = ones
        f0s = torch.stack((ones, c_s_inf), -1)
        return g0s, f0s

    # vectorize index: (0,None,0) index->index[i], dis_ij->dis_ij[i]
    def product(self, index, dis, dis_ij):
        rdis = torch.roll(dis, -index, dims=1)[:, 1:]  # (N, n_conds-1)
        rdis_ij = torch.roll(dis_ij, -index, dims=1)[:, 1:]  # (N, n_conds-1)
        sigma = 0

        denomi = torch.prod(torch.where(
            rdis_ij < 1e-7, torch.tensor(1., dtype=self.dtype), rdis), dim=1)
        numer = torch.prod(torch.where(
            rdis_ij < 1e-7, torch.tensor(1., dtype=self.dtype), rdis_ij), dim=1)
        return denomi/numer  # (N,)

    def connect(self, gx, g0x, f0x):
        # print(gx)
        f_nn = self.forward(gx).view(-1, 1)  # (N, 1)
        # print(f_nn)
        f_g = self.forward(g0x.view(-1, 2))  # (N*n_conds, 1)
        f_g0 = f_g.view(-1, self.ncon_x)  # (N, n_conds)
        delta = self.DELTA_X

        # descriptor distance (N, n_conds)
        dis = torch.sum((gx.view(-1, 1, 2)-g0x)**2, dim=2)
        dis = torch.tanh(dis/delta**2)

        g0x = g0x.view(-1, self.ncon_x, 1, 2)  # (N, n_conds, 1, 2)
        g0xt = torch.permute(g0x, (0, 2, 1, 3))  # (N, 1, n_conds, 2)

        dis_ij = torch.sum((g0x-g0xt)**2, dim=3)  # (N, n_conds, n_conds)
        dis_ij = torch.tanh(dis_ij/delta**2)

        cs = torch.stack(tuple(self.product(i, dis, dis_ij[:, i, :]) for i in range(
            self.ncon_x)), dim=1)  # (N, n_conds)

        fs = (f_nn-f_g0)+f0x  # (N, n_conds)
        total = torch.sum(fs*cs, dim=1)/torch.sum(cs, dim=1)  # (N,)
        return total

    def calc_s0(self, rho, sigma, tau):
        sigma1 = sigma2 = sigma12 = sigma*0.25
        rho01 = rho02 = rho*0.5
        tau1 = tau2 = tau*0.5
        nt = torch.stack((rho01, rho02, sigma1, sigma12,
                          sigma2, tau1, tau2), dim=-1)

        N = nt.shape[0]
        # params = (self.w1, self.w2, self.w3, self.w4,
        #           self.b1, self.b2, self.b3, self.b4)
        # nt=torch.tensor(n, requires_grad=True, dtype=self.dtype)
        gx = self.makeg(nt)[:, 2:]

        g0x, f0x = self.gen_conds(N, gx)

        fx = self.connect(gx, g0x, f0x)
        # print(fx)
        return fx

    def gen_conds_c(self, N, g):
        rho, zeta, s, tau = g[:, 0], g[:, 1], g[:, 2], g[:, 3]

        zeros = torch.zeros(N, dtype=self.dtype)
        ones = torch.ones(N, dtype=self.dtype)

        g0_1 = torch.stack((rho, zeta, zeros, zeros), -1)
        g0_2 = torch.stack((zeros, zeta, s, tau), -1)
        g0_3 = torch.stack((ones, zeta, s, tau), -1)
        g0s = torch.stack((g0_1, g0_2, g0_3), 1)
        self.ncon_c = 3  # g0s.shape[0]

        c_rho_inf = ones

        g_lowdens = torch.stack((rho, zeros, s, tau), -1)
        g_lowdens0 = torch.stack((zeros, zeros, s, tau), -1)

        f_nn_lowdens = self.forward_c(g_lowdens)-self.forward_c(g_lowdens0)+torch.tensor(
            1.0, dtype=self.dtype)  # To make fNN→1 at NNparameters=0

        f0s = torch.stack(
            (ones, f_nn_lowdens[:, 0], c_rho_inf), -1)  # ,c_s_mid
        return g0s, f0s

    def connect_c(self, gc, g0c, f0c):
        f_nn = self.forward_c(gc).view(-1, 1)  # (N, 1)
        f_g = self.forward_c(g0c.view(-1, 4))  # (N*n_conds, 1)
        f_g0 = f_g.view(-1, self.ncon_c)  # (N, n_conds)
        delta = self.DELTA_C

        # descriptor distance (N, n_conds)
        dis = torch.sum((gc.view(-1, 1, 4)-g0c)**2, dim=2)
        dis = torch.tanh(dis/delta**2)

        g0c = g0c.view(-1, self.ncon_c, 1, 4)
        g0ct = torch.permute(g0c, (0, 2, 1, 3))

        dis_ij = torch.sum((g0c-g0ct)**2, dim=3)
        dis_ij = torch.tanh(dis_ij/delta**2)
        cs = torch.stack(tuple(self.product(i, dis, dis_ij[:, i, :]) for i in range(
            self.ncon_c)), dim=1)  # (N, n_conds)

        fs = (f_nn-f_g0)+f0c  # (N, n_conds)
        total = torch.sum(fs*cs, dim=1)/torch.sum(cs, dim=1)  # (N,)
        return total

    def calc_c(self, rho_u, rho_d, sigma_uu, sigma_ud,
               sigma_dd, tau_u, tau_d):

        n = torch.stack((rho_u, rho_d, sigma_uu, sigma_ud,
                         sigma_dd, tau_u, tau_d), dim=-1)
        N = rho_u.shape[0]
        # params_c = (self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4,
        #             self.w1c, self.w2c, self.w3c, self.w4c, self.b1c, self.b2c, self.b3c, self.b4c)

        gc = self.makeg(n)
        g0c, f0c = self.gen_conds_c(N, gc)
        fc = self.connect_c(gc, g0c, f0c)
        # gr = torch.autograd.grad(torch.sum(fc), nt, create_graph=True)
        return fc

    def eval_x(self, densinfo):
        sigma_einsum = "...dr,...dr->...r"
        if not isinstance(densinfo, SpinParam):
            rho = densinfo.value
            grad = densinfo.grad  # (*nrho, ndim)

            # contracted gradient
            sigma = torch.einsum(sigma_einsum, grad, grad)

            lapl = densinfo.lapl
            kin = densinfo.kin

            fx = self.calc_s0(rho, sigma, kin)
            fx = self.shifted_softplus1(fx)
            scan = dqc.get_xc("MGGA_X_SCAN")
            escan = scan.get_edensityxc(densinfo)

            ex = fx*escan  # !!! rho*epsilon
            # print(escan)

        else:
            rho_u = densinfo.u.value
            rho_d = densinfo.d.value

            grad_u = densinfo.u.grad  # (*nrho, ndim)
            grad_d = densinfo.d.grad

            # calculate the contracted gradient
            sigma_uu = torch.einsum(sigma_einsum, grad_u, grad_u)
            sigma_ud = torch.einsum(sigma_einsum, grad_u, grad_d)
            sigma_dd = torch.einsum(sigma_einsum, grad_d, grad_d)

            lapl_u = densinfo.u.lapl
            lapl_d = densinfo.d.lapl
            kin_u = densinfo.u.kin
            kin_d = densinfo.d.kin

            fx1 = self.calc_s0(rho_u*2.0, sigma_uu*4.0, kin_u*2.0)
            fx1 = self.shifted_softplus1(fx1)
            fx2 = self.calc_s0(rho_d*2.0, sigma_dd*4.0, kin_d*2.0)
            fx2 = self.shifted_softplus1(fx2)

            scan = dqc.get_xc("MGGA_X_SCAN")
            densinfo_u2 = ValGrad(
                value=rho_u*2, grad=grad_u*2, lapl=lapl_u*2, kin=kin_u*2)
            densinfo_d2 = ValGrad(
                value=rho_d*2, grad=grad_d*2, lapl=lapl_d*2, kin=kin_d*2)
            escan1 = scan.get_edensityxc(densinfo_u2)
            escan2 = scan.get_edensityxc(densinfo_d2)
            ex = 0.5*(escan1*fx1+escan2*fx2)  # /(rho_u+rho_d)
            #!!! rho*epsilon
        return ex

    def eval_c(self, densinfo):
        sigma_einsum = "...dr,...dr->...r"
        if not isinstance(densinfo, SpinParam):
            rho = densinfo.value
            rho_u = rho_d = rho*0.5

            grad = densinfo.grad  # (*nrho, ndim)

            # contracted gradient
            sigma_uu = sigma_ud = sigma_dd = 0.25 * \
                torch.einsum(sigma_einsum, grad, grad)

            lapl = densinfo.lapl
            kin_u = kin_d = 0.5*densinfo.kin

        else:
            rho_u = densinfo.u.value
            rho_d = densinfo.d.value

            grad_u = densinfo.u.grad  # (*nrho, ndim)
            grad_d = densinfo.d.grad

            # calculate the contracted gradient
            sigma_uu = torch.einsum(sigma_einsum, grad_u, grad_u)
            sigma_ud = torch.einsum(sigma_einsum, grad_u, grad_d)
            sigma_dd = torch.einsum(sigma_einsum, grad_d, grad_d)

            lapl_u = densinfo.u.lapl
            lapl_d = densinfo.d.lapl
            kin_u = densinfo.u.kin
            kin_d = densinfo.d.kin

        fc = self.calc_c(rho_u, rho_d, sigma_uu,
                         sigma_ud, sigma_dd, kin_u, kin_d)
        # gr = np.nan_to_num(gr)
        fc = self.shifted_softplus1(fc)
        scan = dqc.get_xc("MGGA_C_SCAN")
        escan = scan.get_edensityxc(densinfo)
        ec = fc*escan  # !!! rho*epsilon

        return ec

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        ex = self.eval_x(densinfo)
        ec = self.eval_c(densinfo)
        exc = ex+ec

        return exc

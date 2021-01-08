import os
import itertools
import pytest
import torch
from xcdnn2.evaluator import XCDNNEvaluator, PySCFEvaluator
from xcdnn2.dataset import DFTDataset
# from xcdnn2.dft_dataset import XCDNNEvaluator, DFTDataset
from xcdnn2.xcmodels import HybridXC

dtype = torch.float64
filedir = os.path.dirname(os.path.realpath(__file__))
dset_types = ["ie", "ae", "dm", "dens"]

class SimpleNN(torch.nn.Module):
    def __init__(self, w1, w2):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x: (..., nf1)
        # w1: (nf1, nf2)
        # w2: (nf2, nfout)
        z1 = torch.matmul(x, self.w1)  # (..., nf2)
        x1 = self.sigmoid(z1)
        z2 = torch.matmul(x1, self.w2)  # (..., nfout)
        return z2

def get_entry(dset_type):
    # get the first entry of the given dataset type

    # get the dataset of some type of calculation
    fpath = os.path.join(filedir, "test_dataset.yaml")
    dset = DFTDataset(fpath)
    idxs = dset.get_indices(lambda obj: obj["type"] == dset_type)
    if len(idxs) == 0:
        raise RuntimeError("No dataset is selected")
    return dset[idxs[0]]

@pytest.mark.parametrize(
    "dset_type",
    dset_types
)
def test_evaluator_nn_grad(dset_type):
    torch.manual_seed(123)
    entry = get_entry(dset_type)

    weights = {
        "ie": 1.0,
        "ae": 1.0,
        "dm": 1.0,
        "dens": 1.0,
    }
    w1 = torch.nn.Parameter(torch.randn(2, 2, dtype=dtype))
    w2 = torch.nn.Parameter(torch.randn(2, 1, dtype=dtype))

    def get_loss(w1, w2):
        nn = SimpleNN(w1, w2)
        evl = XCDNNEvaluator(HybridXC("lda_x", nn, aweight0=1.0), weights)
        res = evl.calc_loss_function(entry)
        return res

    torch.autograd.gradcheck(get_loss, (w1, w2), eps=1e-3, atol=1e-4)

@pytest.mark.parametrize(
    "dset_type,use_pyscf",
    itertools.product(dset_types, [False, True][1:])
)
def test_evaluator_nn(dset_type, use_pyscf):
    # check the value of get_loss

    # torch.manual_seed(125)  # should not really depend on the random seed
    entry = get_entry(dset_type)

    true_lossval = {
        "ie": 8.7648592243,
        "ae": 1.2724805701,
        "dm": 0.0265032992,
        "dens": 0.2927950191,
    }[dset_type]
    true_lossval = torch.tensor(true_lossval, dtype=dtype)

    # setup the neural network and evaluator
    weights = {
        "ie": 1e3,
        "ae": 1e3,
        "dm": 1e3,
        "dens": 1e3,
    }
    if not use_pyscf:
        w1 = torch.nn.Parameter(torch.randn(2, 2, dtype=dtype))
        w2 = torch.nn.Parameter(torch.randn(2, 1, dtype=dtype))
        nn = SimpleNN(w1, w2)
        evl = XCDNNEvaluator(HybridXC("lda_x", nn, aweight0=0.0), weights)
    else:
        evl = PySCFEvaluator("lda_x", weights)

    # calculate the loss function
    lossval = evl.calc_loss_function(entry)
    # torch.set_printoptions(precision=10)
    # print(lossval)
    assert torch.allclose(lossval, true_lossval)

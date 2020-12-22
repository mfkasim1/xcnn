import pytest
import torch
from xcdnn2.dft_dataset import Evaluator, DFTDataset
from xcdnn2.xcmodels import HybridXC

dtype = torch.float64

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

@pytest.mark.parametrize(
    "dset_type",
    ["ie", "ae", "dm"]
)
def test_evaluator_nn_grad(dset_type):
    torch.manual_seed(123)
    # get the dataset of some type of calculation
    dset = DFTDataset()
    idxs = dset.get_indices(lambda obj: obj["type"] == dset_type)
    if len(idxs) == 0:
        raise RuntimeError("No dataset is selected")

    weights = {
        "ie": 1.0,
        "ae": 1.0,
        "dm": 1.0,
    }
    w1 = torch.nn.Parameter(torch.randn(2, 2, dtype=dtype))
    w2 = torch.nn.Parameter(torch.randn(2, 1, dtype=dtype))

    def get_loss(w1, w2):
        nn = SimpleNN(w1, w2)
        evl = Evaluator(HybridXC("lda_x", nn, aweight0=0.2), weights)
        res = evl.calc_loss_function(dset[idxs[0]])
        return res

    torch.autograd.gradcheck(get_loss, (w1, w2))

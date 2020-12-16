import os
import yaml
from typing import Optional, Dict, Union, List
import torch
import xitorch as xt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from dqc.api.getxc import get_libxc
from dqc.xc.base_xc import BaseXC
from dqc.utils.datastruct import ValGrad, SpinParam
from xcdnn2.dft_dataset import DFTDataset, Evaluator
from xcdnn2.xcmodels import BaseNNXC, NNLDA, NNGGA

###################### training module ######################

class Hybrid(BaseNNXC):
    def __init__(self, xcstr: str, nnmodel: torch.nn.Module,
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

        self.aweight = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True))
        self.bweight = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device, requires_grad=True))
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

class LitDFTXC(pl.LightningModule):
    def __init__(self, evaluator: Evaluator, hparams: Dict):
        super().__init__()
        self.evl = evaluator
        self.hparams = hparams

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.Adam(params, lr=self.hparams["lr"])
        return optimizer

    def training_step(self, train_batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self.evl.calc_loss_function(train_batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, validation_batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self.evl.calc_loss_function(validation_batch)
        self.log("val_loss", loss)
        return loss

if __name__ == "__main__":
    torch.manual_seed(123)
    # hyperparams
    libxc = "gga_x_pbe"
    # libxc = "lda_x"
    nhid = 10
    weights = {
        "ie": 630.0,
        "ae": 630.0,
    }
    hparams = {
        "lr": 1e-4
    }

    family = get_libxc(libxc).family
    if family == 1:
        ninp = 2
    elif family == 2:
        ninp = 3
    else:
        raise RuntimeError("Unimplemented nn for xc family %d" % family)

    torch.autograd.set_detect_anomaly(True)

    # prepare the nn xc model
    nnmodel = torch.nn.Sequential(
        torch.nn.Linear(ninp, nhid),
        torch.nn.Softplus(),
        torch.nn.Linear(nhid, 1, bias=False),
    ).to(torch.double)

    # setup the xc model
    model_nnlda = Hybrid(libxc, nnmodel)
    evl = Evaluator(model_nnlda, weights)
    plsystem = LitDFTXC(evl, hparams)

    # load the dataset and split into train and val
    dset = DFTDataset()
    train_idxs = range(9)
    val_idxs = range(9, 17)
    dset_train = Subset(dset, train_idxs)
    dset_val = Subset(dset, val_idxs)
    dloader_train = DataLoader(dset_train, batch_size=None)
    dloader_val = DataLoader(dset_val, batch_size=None)

    # set up the logger and trainer
    tb_logger = pl.loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(logger=tb_logger)
    trainer.fit(plsystem,
                train_dataloader=dloader_train,
                val_dataloaders=dloader_val)

import argparse
from typing import Dict
import torch
import pytorch_lightning as pl
from dqc.api.getxc import get_libxc
from xcdnn2.xcmodels import HybridXC
from xcdnn2.evaluator import XCDNNEvaluator as Evaluator

# file containing the lightning module and the neural network

###################### training module ######################
class LitDFTXC(pl.LightningModule):
    def __init__(self, hparams: Dict):
        # hparams contains ():
        # * libxc: str
        # * nhid: int
        # * ndepths: int
        # * nn_with_skip: bool
        # * ninpmode: int
        # * outmultmode: int
        # * iew: float
        # * aew: float
        # * dmw: float
        # * densw: float
        super().__init__()

        # handle obsolete option: nnxcmode
        # if specified, then prioritize it over ninpmode and outmultmode and set
        # those parameters according to the value of nnxcmode specified
        nnxcmode = hparams["nnxcmode"]
        if nnxcmode is None:
            pass
        elif nnxcmode == 1:
            hparams["ninpmode"] = 1
            hparams["outmultmode"] = 1
        elif nnxcmode == 2:
            hparams["ninpmode"] = 2
            hparams["outmultmode"] = 2
        elif nnxcmode == 3:
            hparams["ninpmode"] = 2
            hparams["outmultmode"] = 1
        else:
            raise RuntimeError("Invalid value of nnxcmode: %s" % str(nnxcmode))

        self.evl = self._construct_model(hparams)
        self.hparams = hparams

    def _construct_model(self, hparams: Dict) -> Evaluator:
        # model-specific hyperparams
        libxc = hparams["libxc"]
        nhid = hparams["nhid"]
        ndepths = hparams["ndepths"]
        nn_with_skip = hparams.get("nn_with_skip", False)

        # prepare the nn xc model
        family = get_libxc(libxc).family
        if family == 1:
            ninp = 2
        elif family == 2:
            ninp = 3
        else:
            raise RuntimeError("Unimplemented nn for xc family %d" % family)

        # setup the xc nn model
        nnmodel = construct_nn_model(ninp, nhid, ndepths, nn_with_skip).to(torch.double)
        model_nnlda = HybridXC(hparams["libxc"], nnmodel, ninpmode=hparams["ninpmode"],
                               outmultmode=hparams["outmultmode"])

        weights = {
            "ie": hparams["iew"],
            "ae": hparams["aew"],
            "dm": hparams["dmw"],
            "dens": hparams["densw"],
        }
        self.dweights = {  # weights from the dataset
            "ie": 440.0,
            "ae": 1340.0,
            "dm": 220.0,
            "dens": 170.0,
        }
        self.weights = weights
        self.type_indices = {x: i for i, x in enumerate(self.weights.keys())}
        return Evaluator(model_nnlda, weights)

    def configure_optimizers(self):
        params = list(self.parameters())

        # making optimizer for every type of datasets (to stabilize the gradients)
        opts = [torch.optim.Adam(params, lr=self.hparams["%slr" % tpe]) for tpe in self.weights]
        return opts

    def forward(self, x: Dict) -> torch.Tensor:
        return self.evl.calc_loss_function(x)

    def training_step(self, train_batch: Dict, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        # obtain which optimizer should be performed based on the batch type
        tpe = train_batch["type"]
        if self.hparams["split_opt"]:
            idx = self.type_indices[tpe]
        else:
            idx = 0
        opt = self.optimizers()[idx]

        # perform the backward pass manually
        loss = self.forward(train_batch)
        self.manual_backward(loss, opt)
        opt.step()
        opt.zero_grad()

        # log the training loss
        self.log("train_loss_%s" % tpe, loss.detach(), on_step=False, on_epoch=True)
        self.log("train_loss", loss.detach(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, validation_batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self.forward(validation_batch)

        tpe = validation_batch["type"]
        rawloss = loss.detach() / self.weights[tpe]  # raw loss without weighting
        vloss = rawloss * self.dweights[tpe]  # normalized loss standardized by the datasets' mean
        self.log("val_loss", vloss, on_step=False, on_epoch=True)
        self.log("val_loss_%s" % validation_batch["type"], rawloss, on_step=False, on_epoch=True)

        return loss

    @staticmethod
    def get_trainer_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # params that are specific to the model
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # arguments to be stored in the hparams file
        # model hyperparams
        parser.add_argument("--nhid", type=int, default=10,
                            help="The number of elements in hidden layers")
        parser.add_argument("--ndepths", type=int, default=1,
                            help="The number of hidden layers depths")
        parser.add_argument("--nn_with_skip", action="store_const", const=True, default=False,
                            help="Add skip connection in the neural network")
        parser.add_argument("--libxc", type=str, default="lda_x",
                            help="Initial xc to be used")
        parser.add_argument("--ninpmode", type=int, default=1,
                            help="The mode to decide the transformation of density to the NN input")
        parser.add_argument("--outmultmode", type=int, default=1,
                            help="The mode to decide the Eks from NN output")
        parser.add_argument("--nnxcmode", type=int,
                            help="The mode to decide how to compute Exc from NN output (obsolete, do not use)")

        # hparams for the loss function
        parser.add_argument("--iew", type=float, default=440.0,
                            help="Weight of ionization energy")
        parser.add_argument("--aew", type=float, default=1340.0,
                            help="Weight of atomization energy")
        parser.add_argument("--dmw", type=float, default=220.0,
                            help="Weight of density matrix")
        parser.add_argument("--densw", type=float, default=170.0,
                            help="Weight of density profile loss")

        # hparams for optimizer
        parser.add_argument("--split_opt", action="store_const", default=False, const=True,
                            help="Flag to split optimizer based on the dataset type")
        parser.add_argument("--ielr", type=float, default=1e-4,
                            help="Learning rate for ionization energy (chosen if there is --split_opt)")
        parser.add_argument("--aelr", type=float, default=1e-4,
                            help="Learning rate for atomization energy (ignored if no --split_opt)")
        parser.add_argument("--dmlr", type=float, default=1e-4,
                            help="Learning rate for density matrix (ignored if no --split_opt)")
        parser.add_argument("--denslr", type=float, default=1e-4,
                            help="Learning rate for density profile (ignored if no --split_opt)")
        return parser

class NNModel(torch.nn.Module):
    def __init__(self, ninp: int, nhid: int, ndepths: int, with_skip: bool = False):
        super().__init__()
        layers = []
        activations = []
        if with_skip:
            skip_weights = []
            conn_weights = []
        for i in range(ndepths):
            n1 = ninp if i == 0 else nhid
            layers.append(torch.nn.Linear(n1, nhid))
            activations.append(torch.nn.Softplus())
            if with_skip and i >= 1:
                # using Linear instead of parameter to avoid userwarning
                # of parameterlist not supporting set attributes
                conn_weights.append(torch.nn.Linear(1, 1, bias=False))
                skip_weights.append(torch.nn.Linear(1, 1, bias=False))
                # conn_weights.append(torch.nn.Parameter(torch.tensor(0.5)))
                # skip_weights.append(torch.nn.Parameter(torch.tensor(0.5)))
        layers.append(torch.nn.Linear(nhid, 1, bias=False))

        # construct the nn parameters
        self.layers = torch.nn.ModuleList(layers)
        self.activations = torch.nn.ModuleList(activations)
        self.with_skip = with_skip
        if with_skip:
            self.conn_weights = torch.nn.ModuleList(conn_weights)
            self.skip_weights = torch.nn.ModuleList(skip_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)):
            y = self.layers[i](x)

            # activation (no activation at the last layer)
            if i < len(self.activations):
                y = self.activations[i](y)

            # skip connection (no skip at the first and last layer)
            if self.with_skip and i >= 1 and i < len(self.layers) - 1:
                y1 = self.conn_weights[i - 1](y.unsqueeze(-1))
                y2 = self.skip_weights[i - 1](x.unsqueeze(-1))
                y = (y1 + y2).squeeze(-1)
            x = y
        return x

def construct_nn_model(ninp: int, nhid: int, ndepths: int, with_skip: bool = False):
    # construct the neural network model of the xc energy
    if not with_skip:
        # old version, to enable loading the old models
        layers = []
        for i in range(ndepths):
            n1 = ninp if i == 0 else nhid
            layers.append(torch.nn.Linear(n1, nhid))
            layers.append(torch.nn.Softplus())
        layers.append(torch.nn.Linear(nhid, 1, bias=False))
        return torch.nn.Sequential(*layers)
    else:
        return NNModel(ninp, nhid, ndepths, with_skip)

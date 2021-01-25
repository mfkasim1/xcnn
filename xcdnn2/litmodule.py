import argparse
import warnings
from typing import Dict, List
import torch
import pytorch_lightning as pl
from dqc.api.getxc import get_xc
from xcdnn2.xcmodels import HybridXC
from xcdnn2.evaluator import XCDNNEvaluator, PySCFEvaluator

# file containing the lightning module and the neural network

###################### training module ######################
class LitDFTXC(pl.LightningModule):
    def __init__(self, hparams: Dict, entries: List[Dict] = []):
        # hparams contains ():
        # * libxc: str
        # * nhid: int
        # * ndepths: int
        # * nn_with_skip: bool
        # * ninpmode: int
        # * sinpmode: int
        # * outmultmode: int
        # * iew: float
        # * aew: float
        # * dmw: float
        # * densw: float
        super().__init__()

        # handle deprecated option: nnxcmode
        # if specified, then prioritize it over ninpmode and outmultmode and set
        # those parameters according to the value of nnxcmode specified
        nnxcmode = hparams.get("nnxcmode", None)
        if nnxcmode is not None:
            warnings.warn("--nnxcmode flag is deprecated, please use --ninpmode and --outmultmode")
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

        self.evl = self._construct_model(hparams, entries)
        self.hparams = hparams

    def _construct_model(self, hparams: Dict, entries: List[Dict] = []) -> XCDNNEvaluator:

        # set the weights
        weights = {
            "ie": hparams.get("iew", 440.),
            "ae": hparams.get("aew", 1340.),
            "dm": hparams.get("dmw", 220.),
            "dens": hparams.get("densw", 170.),
        }
        self.dweights = {  # weights from the dataset
            "ie": 440.0,
            "ae": 1340.0,
            "dm": 220.0,
            "dens": 170.0,
        }
        self.weights = weights
        self.type_indices = {x: i for i, x in enumerate(self.weights.keys())}

        self.use_pyscf = hparams.get("pyscf", False)
        if not self.use_pyscf:
            # prepare the nn xc model
            libxc_dqc = hparams["libxc"].replace(",", "+")
            family = get_xc(libxc_dqc).family
            if family == 1:
                ninp = 2
            elif family == 2:
                ninp = 3
            else:
                raise RuntimeError("Unimplemented nn for xc family %d" % family)

            # setup the xc nn model
            nhid = hparams["nhid"]
            ndepths = hparams["ndepths"]
            nn_with_skip = hparams.get("nn_with_skip", False)

            nnmodel = construct_nn_model(ninp, nhid, ndepths, nn_with_skip).to(torch.double)
            model_nnlda = HybridXC(hparams["libxc"], nnmodel,
                                   ninpmode=hparams["ninpmode"],
                                   sinpmode=hparams.get("sinpmode", 1),
                                   outmultmode=hparams["outmultmode"])
            always_attach = hparams.get("always_attach", False)
            return XCDNNEvaluator(model_nnlda, weights,
                                  always_attach=always_attach,
                                  entries=entries)
        else:
            # if using pyscf, no neural network is constructed
            # dummy parameter required just to make it run without error
            self.dummy_param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double))
            return PySCFEvaluator(hparams["libxc"], weights)

    def configure_optimizers(self):
        params = list(self.parameters())

        # making optimizer for every type of datasets (to stabilize the gradients)
        opt_str = self.hparams.get("optimizer", "adam").lower()
        if opt_str == "adam":
            opt_cls = torch.optim.Adam
        elif opt_str == "radam":
            from radam import RAdam
            opt_cls = RAdam
        else:
            raise RuntimeError("Unknown optimizer %s" % opt_str)
        wdecay = self.hparams.get("wdecay", 0.0)
        opts = [opt_cls(params, lr=self.hparams["%slr" % tpe], weight_decay=wdecay) for tpe in self.weights]
        return opts

    def forward(self, x: Dict) -> torch.Tensor:
        res = self.evl.calc_loss_function(x)
        if self.use_pyscf:
            res = res + self.dummy_param * 0
        return res

    def deviation(self, x: Dict) -> torch.Tensor:
        # deviation of the predicted value and true value in a meaningful format
        return self.evl.calc_deviation(x)

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
        parser.add_argument("--sinpmode", type=int, default=1,
                            help="The mode to decide the transformation of normalized grad density to the NN input")
        parser.add_argument("--outmultmode", type=int, default=1,
                            help="The mode to decide the Eks from NN output")
        parser.add_argument("--nnxcmode", type=int,
                            help="The mode to decide how to compute Exc from NN output (deprecated, do not use)")
        parser.add_argument("--pyscf", action="store_const", const=True, default=False,
                            help="Using pyscf calculation. If activated, the nn-related arguments are ignored.")
        parser.add_argument("--always_attach", action="store_const", const=True, default=False,
                            help="Always propagate gradient even if the iteration does not converge")

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
        parser.add_argument("--optimizer", type=str, default="adam",
                            help="Optimizer algorithm")
        parser.add_argument("--wdecay", type=float, default=0.0,
                            help="Weight decay of the algorithm (i.e. L2 regularization)")
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

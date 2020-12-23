from typing import Dict
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from dqc.api.getxc import get_libxc
from xcdnn2.dft_dataset import DFTDataset, Evaluator
from xcdnn2.xcmodels import HybridXC

###################### training module ######################
class LitDFTXC(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super().__init__()
        self.evl = self._construct_model(hparams)
        self.hparams = hparams

    def _construct_model(self, hparams: Dict) -> Evaluator:
        # model-specific hyperparams
        libxc = hparams["libxc"]
        nhid = hparams["nhid"]

        # prepare the nn xc model
        family = get_libxc(libxc).family
        if family == 1:
            ninp = 2
        elif family == 2:
            ninp = 3
        else:
            raise RuntimeError("Unimplemented nn for xc family %d" % family)

        # setup the xc nn model
        nnmodel = torch.nn.Sequential(
            torch.nn.Linear(ninp, nhid),
            torch.nn.Softplus(),
            torch.nn.Linear(nhid, 1, bias=False),
        ).to(torch.double)
        model_nnlda = HybridXC(args.libxc, nnmodel)

        weights = {
            "ie": hparams["iew"],
            "ae": hparams["aew"],
            "dm": hparams["dmw"],
        }
        self.dweights = {  # weights from the dataset
            "ie": 1340.0,
            "ae": 440.0,
            "dm": 220.0,
        }
        self.weights = weights
        self.type_indices = {x: i for i, x in enumerate(self.weights.keys())}
        return Evaluator(model_nnlda, weights)

    def configure_optimizers(self):
        params = list(self.parameters())

        # making optimizer for every type of datasets (to stabilize the gradients)
        opts = [torch.optim.Adam(params, lr=self.hparams["lr"]) for tpe in self.weights]
        return opts

    def training_step(self, train_batch: Dict, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        # obtain which optimizer should be performed based on the batch type
        tpe = train_batch["type"]
        if self.hparams["split_opt"]:
            idx = self.type_indices[tpe]
        else:
            idx = 0
        opt = self.optimizers()[idx]

        # perform the backward pass manually
        loss = self.evl.calc_loss_function(train_batch)
        self.manual_backward(loss, opt)
        opt.step()
        opt.zero_grad()

        # log the training loss
        self.log("train_loss", loss.detach(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, validation_batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self.evl.calc_loss_function(validation_batch)

        tpe = validation_batch["type"]
        rawloss = loss.detach() / self.weights[tpe]  # raw loss without weighting
        vloss = rawloss * self.dweights[tpe]  # normalized loss standardized by the datasets' mean
        self.log("val_loss", vloss, on_step=False, on_epoch=True)
        self.log("val_loss_%s" % validation_batch["type"], rawloss, on_step=False, on_epoch=True)

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # arguments to be stored in the hparams file
        parser.add_argument("--nhid", type=int, default=10,
                            help="The number of hidden layers")
        parser.add_argument("--libxc", type=str, default="lda_x",
                            help="Initial xc to be used")
        parser.add_argument("--lr", type=float, default=1e-4,
                            help="Learning rate")
        parser.add_argument("--clipval", type=float, default=0,
                            help="Clip gradients with norm above this value. 0 means no clipping.")
        parser.add_argument("--iew", type=float, default=440.0,
                            help="Weight of ionization energy")
        parser.add_argument("--aew", type=float, default=1340.0,
                            help="Weight of atomization energy")
        parser.add_argument("--dmw", type=float, default=220.0,
                            help="Weight of density matrix")
        parser.add_argument("--tvset", type=int, default=2,
                            help="Training/validation set")
        parser.add_argument("--exclude_types", type=str, nargs="*", default=[],
                            help="Exclude several types of dataset")
        parser.add_argument("--split_opt", action="store_const", default=False, const=True,
                            help="Flag to split optimizer based on the dataset type")
        return parser

if __name__ == "__main__":
    from xcdnn2.utils import subs_present

    torch.manual_seed(123)

    # parsing the hyperparams
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_const", default=False, const=True,
                        help="Flag to record the progress")
    parser = LitDFTXC.add_model_specific_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # putting all the hyperparameters in a dictionary
    hparams = vars(args)

    # create the lightning module
    plsystem = LitDFTXC(hparams)

    # load the dataset and split into train and val
    dset = DFTDataset()
    if args.tvset == 1:
        # train_atoms = ["H", "He", "Li", "Be", "B", "C"]
        val_atoms = ["N", "O", "F", "Ne"]
    elif args.tvset == 2:  # randomly selected
        # train_atoms = ["H", "Li", "B", "C", "O", "Ne"]
        val_atoms = ["He", "Be", "N", "F"]

    general_filter = lambda obj: obj["type"] not in args.exclude_types
    all_idxs = dset.get_indices(general_filter)
    val_filter = lambda obj: subs_present(val_atoms, obj["name"].split()[-1]) and general_filter(obj)
    val_idxs = dset.get_indices(val_filter)
    train_idxs = list(set(all_idxs) - set(val_idxs))
    # print(train_idxs, len(train_idxs))
    # print(val_idxs, len(val_idxs))
    # raise RuntimeError

    # train_idxs = range(24, 26)  # dm
    # val_idxs = range(20, 23)
    # train_idxs = range(3)  # ie
    # val_idxs = range(3, 6)
    dset_train = Subset(dset, train_idxs)
    dset_val = Subset(dset, val_idxs)
    dloader_train = DataLoader(dset_train, batch_size=None)
    dloader_val = DataLoader(dset_val, batch_size=None)

    # setup the trainer
    trainer_kwargs = {
        "logger": False,
        "checkpoint_callback": False,
        "num_sanity_val_steps": 0,
        "gradient_clip_val": args.clipval,
        "automatic_optimization": False,
    }
    if args.record:
        # set up the logger
        tb_logger = pl.loggers.TensorBoardLogger('logs/')
        chkpt_val = ModelCheckpoint(monitor="val_loss", save_top_k=4, save_last=True)
        trainer_kwargs["logger"] = tb_logger
        trainer_kwargs["callbacks"] = [chkpt_val]
        trainer_kwargs["checkpoint_callback"] = True

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(plsystem,
                train_dataloader=dloader_train,
                val_dataloaders=dloader_val)

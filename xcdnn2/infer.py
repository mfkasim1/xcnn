import argparse
from typing import List, Dict
import torch
import matplotlib.pyplot as plt
from xcdnn2.litmodule import LitDFTXC
from xcdnn2.dataset import DFTDataset

class Plotter(object):
    # this object provides interface to plot the losses
    def __init__(self, ntypes: int, hparams: Dict):
        self.losses: List[List[float]] = []  # self.losses[i_entry][i_model]
        self.hparams = self._set_default_hparams(ntypes, hparams)
        self.ntypes = ntypes

    def add_losses(self, losses: List[float]):
        # add the losses from different types of models for one entry
        assert len(losses) == self.ntypes
        self.losses.append(losses)

    def show(self):
        # show the plot of the losses in the current axes
        assert len(self.losses) > 0
        plt.plot(self.losses, 'o')
        if hparams["labels"]:
            plt.legend(hparams["labels"])
        if hparams["title"]:
            plt.title(hparams["title"])
        if hparams["xlabel"]:
            plt.xlabel(hparams["xlabel"])
        if hparams["ylabel"]:
            plt.ylabel(hparams["ylabel"])
        plt.show()

    def _set_default_hparams(self, ntypes: int, hparams: Dict):
        # set the default hparams
        # currently there's nothing to do here
        pass

    @staticmethod
    def get_plot_argparse(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--labels", type=str, nargs="*",
                            help="Labels for the checkpoints")
        parser.add_argument("--title", type=str,
                            help="Title of the plot")
        parser.add_argument("--ylabel", type=str,
                            help="y-axis label of the plot")
        parser.add_argument("--xlabel", type=str,
                            help="x-axis label of the plot")
        return parser

def get_infer_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="The dataset where the inference is taking place")
    parser.add_argument("--chkpts", type=str, nargs="+",
                        help="Checkpoints where the models are loaded from")

    # plot options
    parser.add_argument("--plot", action="store_const", default=False, const=True,
                        help="If present, plot the values")
    return parser

if __name__ == "__main__":
    parser = get_infer_argparse()
    parser = Plotter.get_plot_argparse(parser)
    # parser = LitDFTXC.get_trainer_argparse(parser)
    args = parser.parse_args()

    # putting all the hyperparameters in a dictionary
    hparams = vars(args)

    # load the model and the dataset
    models = [LitDFTXC.load_from_checkpoint(checkpoint_path=chkpt) for chkpt in hparams["chkpts"]]
    dset = DFTDataset(hparams["dataset"])
    plotter = Plotter(len(hparams["chkpts"]), hparams)

    # calculate the losses for all entries and models
    for i in range(len(dset)):
        losses = [float(model.deviation(dset[i]).item()) for model in models]
        losses_str = ", ".join(["%.4e" % loss for loss in losses])
        print("%d out of %d: %s (%s)" % (i + 1, len(dset), dset[i]["name"], losses_str))
        plotter.add_losses(losses)

    # show the plot
    if hparams["plot"]:
        plotter.show()

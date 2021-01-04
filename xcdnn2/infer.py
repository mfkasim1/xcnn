import argparse
from typing import List, Dict
import torch
import matplotlib.pyplot as plt
from xcdnn2.litmodule import LitDFTXC
from xcdnn2.dataset import DFTDataset

class Plotter(object):
    # this object provides interface to plot the losses
    def __init__(self, ntypes: int, hparams: Dict, all_losses: List[List[float]]):
        self.losses = all_losses  # self.losses[i_entry][i_model]
        self.hparams = self._set_default_hparams(ntypes, hparams)
        self.ntypes = ntypes

    def show(self):
        # show the plot of the losses in the current axes
        assert len(self.losses) > 0
        plt.plot(self.losses, 'o')
        if self.hparams["labels"]:
            plt.legend(self.hparams["labels"])
        if self.hparams["title"]:
            plt.title(self.hparams["title"])
        if self.hparams["xlabel"]:
            plt.xlabel(self.hparams["xlabel"])
        if self.hparams["ylabel"]:
            plt.ylabel(self.hparams["ylabel"])
        plt.show()

    def _set_default_hparams(self, ntypes: int, hparams: Dict):
        # set the default hparams
        # currently there's nothing to do here
        return hparams

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

def list2str(x: List[float], fmt: str = "%.4e", sep: str = ", ") -> str:
    # convert a list of float into a string
    return sep.join([fmt % xx for xx in x])

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

    # calculate the losses for all entries and models
    all_losses = []
    for i in range(len(dset)):
        losses = [float(model.deviation(dset[i]).item()) for model in models]
        losses_str = list2str(losses)
        print("%d out of %d: %s (%s)" % (i + 1, len(dset), dset[i]["name"], losses_str))
        all_losses.append(losses)

    # get the mean
    all_losses = np.array(all_losses)
    print("     Mean absolute error (MAE): %s" % list2str(np.mean(all_losses, axis=0)))
    print("Root mean squared error (RMSE): %s" % list2str(np.sqrt(np.mean(all_losses ** 2, axis=0))))

    # show the plot
    if hparams["plot"]:
        plotter = Plotter(len(hparams["chkpts"]), hparams, all_losses)
        plotter.show()

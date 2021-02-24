import os
import argparse
from typing import List, Dict, Optional, Union
import torch
import numpy as np
from pysr import pysr, best
from xcdnn2.litmodule import LitDFTXC
from xcdnn2.xcmodels import HybridXC

# symbolic regression of the learned xc model with pysr
# NOTE: this file only learns the neural network part, without the libxc part

def generate_param(param: str, nsize: int) -> np.ndarray:
    # generate parameters
    if param == "n":
        return np.exp(np.random.random(nsize) * 2.4) - 1  # from 0 to about 10
    elif param == "xi":
        return np.random.random(nsize)
    elif param == "s":
        return np.exp(np.random.random(nsize) * 2.4) - 1
    else:
        raise RuntimeError("Unknown param: %s" % param)

def get_symreg_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("chkpt", type=str,
                        help="Checkpoint where the model is loaded from")
    parser.add_argument("--showparams", action="store_const", default=False, const=True,
                        help="If enabled, then show the parameters of the loaded checkpoint")
    parser.add_argument("--nsize", type=int, default=100,
                        help="Total size of the training data")
    return parser

if __name__ == "__main__":
    parser = get_symreg_argparse()
    args = parser.parse_args()

    # load the model
    model = LitDFTXC.load_from_checkpoint(checkpoint_path=args.chkpt, strict=False)
    if args.showparams:
        print(list(model.parameters()))

    # get the xc object
    xc = model.evl.get_xc()
    family = xc.family
    # the function to be searched for its symbolic model
    assert isinstance(xc, HybridXC)
    fx = xc.nnxc.nnmodel  # assuming xc is HybridXC

    # generate the sample data
    nsize = args.nsize
    params = []
    if family >= 1:  # LDA or above
        params.append(generate_param("n", nsize))
        params.append(generate_param("xi", nsize))
    if family >= 2:  # GGA or above
        params.append(generate_param("s", nsize))
    if family > 2 or family < 1:
        raise RuntimeError("xc family %d has not been implemented" % family)
    X = np.concatenate(params).reshape((len(params), -1)).T  # (ndata, nfeats)
    y = fx(torch.as_tensor(X)).view(-1).detach().numpy()

    binary_ops = ["plus", "sub", "mult", "pow", "div"]
    unary_ops = ["neg", "square", "cube", "exp", "abs", "logm", "sqrtm",
                 "log1pp(x)=log(1+abs(x))",
                 "sin", "cos", "tan", "sinh", "cosh", "tanh",
                 # "asin", "acos",
                 "atan", "asinh",
                 # "acosh", "atanh",
                 ]
    equations = pysr(X, y, niterations=5,
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        maxsize=30)
    print(best(equations))

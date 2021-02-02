import argparse
import numpy as np
from typing import Tuple, List

def input_parser():
    # parse the input from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, help="The results file name")
    parser.add_argument("--calcsubset", action="store_const", default=False, const=True,
                        help="If enabled, then calculate the statistics for the subset of the entries")
    args = parser.parse_args()
    return args

def parse_file(fname: str) -> Tuple[List[str], List[str], np.ndarray]:
    # parse the results file and returns:
    # * list of checkpoints
    # * list of entry names
    # * entry values in a 2D numpy array
    chkpts = []
    chkpt_marker = "# Checkpoints:"
    entry_marker = "out of"

    entries = []
    names = []
    with open(fname, "r") as f:
        for line in f:
            # check point lines
            if line.startswith(chkpt_marker):
                chkpts = line[len(chkpt_marker):].strip().split("|")
            # entry lines
            elif entry_marker in line:
                components = line.split(":")
                vals = eval(components[-1].strip())
                if not isinstance(vals, tuple):
                    vals = tuple([vals])
                entries.append(vals)
                names.append(components[1].strip())
            else:
                pass
    return chkpts, names, np.array(entries)

def np2str(a: np.ndarray, sep: str = ", ", fmt: str = "%.4e") -> str:
    # convert the numpy 1D array to a string
    return sep.join([fmt % aa for aa in a])

def print_stats(values: np.ndarray):
    # print the error statistics of a 2D numpy array

    # check if any non-convergence calculation (here defined as error > 500)
    threshold = 5e2
    vals = np.copy(values)
    nonconv = np.abs(vals) > threshold
    vals[nonconv] = float("nan")
    if np.any(nonconv):
        print("Non convergence here")

    # print the stats without the non-convergence results
    print("  ME: %s" % np2str(np.nanmean(vals, axis=0)))
    print(" MAE: %s" % np2str(np.nanmean(np.abs(vals), axis=0)))
    print("RMSE: %s" % np2str(np.nanmean(vals * vals, axis=0) ** 0.5))

def get_subsets(entry_names: List[str], values: np.ndarray) -> Tuple[List[str], List[List[str]], List[np.ndarray]]:
    # split the dataset into various subsets, then returns:
    # * list of the subset names (e.g. hydrocarbons)
    # * list of a list of entry names in each subset
    # * list of 2D numpy arrays for each subset
    # NOTE: This function is specific for each dataset, so any changes in the
    # dataset must be reflected in this function

    # Gauss2 atomization energy dataset
    if len(entry_names) == 110:
        subset_names = ["hydrocarbons", "subs-hydrocarbons", "other-1", "other-2"]
        indices = [
            [2, 3, 4, 5, 21, 22, 23, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 107],
            [29, 49, 50, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 108],
            [0, 1, 6, 7, 8, 9, 10, 11, 19, 20, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 53, 57, 61, 64, 66, 67, 68, 69, 70, 105, 109],
            [12, 13, 14, 15, 16, 17, 18, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 65, 106],
        ]
    # ionization energy
    elif len(entry_names) == 18:
        subset_names = ["trainval", "other"]
        indices = [
            [6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17],
        ]
    else:
        raise RuntimeError("Unknown dataset with %d entries" % len(entry_names))

    # get the subset of the entry names and the values
    svalues = []
    sentrynames = []
    for idx in indices:
        svalues.append(values[np.array(idx), :])
        sentrynames.append([entry_names[i] for i in idx])
    return subset_names, sentrynames, svalues

def main():
    args = input_parser()
    chkpts, names, values = parse_file(args.fname)

    # calculate the error statistics for all groups
    print("Checkpoints: %s" % "|".join(chkpts))
    print_stats(values)

    if args.calcsubset:
        subset_names, entryname_subsets, val_subsets = get_subsets(names, values)
        for sname, vals in zip(subset_names, val_subsets):
            print("----------------------------")
            print("Subset: %s" % sname)
            print_stats(vals)

if __name__ == "__main__":
    main()

from typing import Tuple, List
import argparse
import numpy as np
import matplotlib.pyplot as plt

def input_parser():
    # parse the input from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, help="The results file name")
    args = parser.parse_args()
    return args

def parse_file(fname: str) -> Tuple[List[str], List[str], np.ndarray]:
    # parse file containing the results in csv
    # returns: (1) list of headers, (2) list of groups of the entries,
    #          (3) the values per entries in 2D numpy array
    delim = ","
    group_col = 1
    # load the header
    headers = list(np.loadtxt(fname, delimiter=delim, max_rows=1, dtype=str))
    ncols = len(headers)
    groups = list(np.loadtxt(fname, delimiter=delim, skiprows=1, dtype=str, usecols=[group_col]).ravel())
    content = np.abs(np.loadtxt(fname, delimiter=delim, skiprows=1, usecols=list(range(group_col + 1, ncols))))
    return headers[2:], groups, content

def main():
    # parameters to be set by the user
    selected_headers = [
        "PBE",
        "XCNN-PBE",
        "XCNN-PBE-IP",
        "CCSD (cc-pvqz)",
        "CCSD-T (cc-pvqz)",
    ]
    header_colors = ["C%d" % i for i in range(len(selected_headers))]
    group_order = {
        "IP 18": lambda g: g == "IP 18",
        "AE 104": lambda g: g.startswith("AE"),
        "AE 16 HC": lambda g: g == "AE 16 HC",
        "AE 25 subs HC": lambda g: g == "AE 25 subs HC",
        "AE 33 others-1": lambda g: g == "AE 33 others-1",
        "AE 30 others-2": lambda g: g == "AE 30 others-2",
    }

    # plot parameters
    fontsize = 11
    group_spacing = 3  # separation of each group
    figsize = (6, 3)
    show_outliers = False

    ngroups = len(group_order)
    args = input_parser()
    headers, groups, content = parse_file(args.fname)

    # obtain the header idxs
    header_idxs = np.array([headers.index(sh) for sh in selected_headers])
    print(headers)

    # get the data that belongs to each group
    all_data = []
    for group, fcn in group_order.items():
        group_idx = np.array([fcn(g) for g in groups])
        data = content[group_idx, :]
        data = data[:, header_idxs].T  # (nheaders, ndata)
        all_data.append([d for d in data])

    # all_data: (ngroup, nheaders, ndata)
    # transpose the data
    all_data = list(zip(*all_data))
    # all_data: (nheaders, ngroup, ndata)

    # separation between the beginning of the group and the next group
    group_sep = group_spacing + len(selected_headers)

    # middle index of each group
    imid = (len(selected_headers) - 1) // 2

    plt.figure(figsize=figsize)
    bpobjs = []  # boxplot objects
    for i, group_data in enumerate(all_data):
        # group_data: (ngroup, ndata)
        color = header_colors[i]
        bp = plt.boxplot(group_data,
                         positions = np.arange(ngroups) * group_sep + i,
                         showfliers = show_outliers,
                         boxprops = dict(color=color),
                         capprops = dict(color=color),
                         whiskerprops = dict(color=color),
                         medianprops = dict(color=color),
                         flierprops = dict(markeredgecolor=color, markersize=2),
                         )
        bpobjs.append(bp["boxes"][0])  # only save the first box
    # plt.gca().set_yscale("log")
    plt.legend(bpobjs, selected_headers)
    plt.xticks(ticks=np.arange(ngroups) * group_sep + imid,
               labels=group_order.keys(),
               rotation=10,
               fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel("Absolute error (kcal/mol)", fontsize=fontsize)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

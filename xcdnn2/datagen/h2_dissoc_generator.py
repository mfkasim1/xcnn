import yaml
import numpy as np
from typing import List, Dict
from pyscf import cc, scf, gto
from xcdnn2.datagen.retriever import System

def get_h2_dissoc_entries(dists: List[float], basis: str = "6-311++G**") -> List[Dict]:
    # get entries for dissociation curve

    all_res = []
    for i, dist in enumerate(dists):
        print("%d out of %d: %.4f" % (i + 1, len(dists), dist))
        moldesc = "H 0 0 0; H 0 0 %.4f" % dist

        res = {}
        res["name"] = "Total energy of H2 at %.4f" % dist
        res["type"] = "ae"
        res["true_val"] = get_ccsd_energy(moldesc, basis) + 1.0

        # construct the command
        res["cmd"] = "energy(systems[0]) - 2 * energy(systems[1])"

        # construct the system
        res["systems"] = [
            System.create("mol", moldesc, numel0=2, basis=basis),
            System.create("mol", "H 0 0 0", numel0=1, basis=basis),
        ]
        all_res.append(res)
    return all_res

def get_ccsd_energy(moldesc: str, basis: str) -> float:
    # get the H2 energy at the given distance using CCSD
    mol = gto.M(atom=moldesc, basis=basis, unit="Bohr")
    mqc = scf.RHF(mol).run()
    qc = cc.RCCSD(mqc)
    qc.kernel()
    return float(qc.e_tot)

if __name__ == "__main__":
    dists = np.linspace(0.5, 4, 20)
    entries = get_h2_dissoc_entries(dists)
    with open("h2_dissoc_curve.yaml", "w") as f:
        f.write(yaml.dump(entries, sort_keys=False))

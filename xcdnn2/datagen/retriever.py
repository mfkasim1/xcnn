from __future__ import annotations
import requests
import copy
import re
import yaml
from collections import Counter
from typing import List
from pyscf import gto, dft
from bs4 import BeautifulSoup
from dqc.utils.periodictable import get_atomz
from xcdnn2.datagen.utils import angstrom2bohr, get_atom_dHf0, energy2hartree

class AtomConf(object):
    ################# crawler #################
    @staticmethod
    def _get_atompos_cccbdb(soup) -> Tuple[List[str], List[List[float]]]:
        # retrieve the atom type and the positions from the cccbdb table

        # get the cartesian table
        table_found = False
        for sp in soup.find_all("span"):
            if sp.get_text().strip() == "Cartesians":
                table_found = True
                cart_table = sp.findNext("table")
        if not table_found:
            raise RuntimeError("The Cartesians table is not found")

        # read the rows of the cartesian table
        atoms = []
        all_poss = []
        pattern = r"^([A-Za-z]+)[0-9]*"
        for i, row in enumerate(cart_table.find_all("tr")):
            if i == 0:
                continue  # skip the first row
            pos = []
            for j, col in enumerate(row.find_all("td")):
                if j == 0:  # atom name
                    m = re.search(pattern, col.get_text())
                    atoms.append(m.group(1))
                else:
                    # atom position (directly convert to Bohr unit)
                    p = angstrom2bohr(float(col.get_text()))
                    pos.append(p)
            all_poss.append(pos)

        assert len(atoms) == len(all_poss)
        assert len(atoms) > 0
        assert len(all_poss[0]) == 3
        return atoms, all_poss

    @staticmethod
    def _get_enthalpy(soup) -> Tuple[Optional[float], Optional[float]]:
        # retrieve the enthalpy of formations at 0K and 298K

        def __get_enthalpy_val(s: str) -> Optional[float]:
            try:
                td = soup.find(text=re.compile(s)).parent
                val_obj = td.findNext("td")
                val = float(val_obj.get_text())
                unit = val_obj.findNext("td").findNext("td").get_text()
                # convert the energy to Hartree
                val: Optional[float] = energy2hartree(val, unit)
            except:
                val = None
            return val

        val0 = __get_enthalpy_val(r"Hfg\(0K\)")
        val298 = __get_enthalpy_val(r"Hfg\(298\.15K\)")
        return val0, val298

    @staticmethod
    def _get_zpe(soup) -> Optional[float]:
        # retrieve the zero point energy
        th = soup.find(text="ZPE")
        val: Optional[float] = None
        if th is not None:
            # retrieve from the experimental data
            tr = th.parent.parent.parent
            tr2 = tr.findNext("tr")
            for i, td in enumerate(tr2.find_all("td")):
                if i != 5:
                    continue
                val = float(td.get_text())
                val = energy2hartree(val, "cm-1")
        else:
            # retrieve the zero point energy from the fundamental vibration
            elmt = soup.find(text=re.compile("vibrational zero-point energy"))
            if elmt is None:
                return None
            txt = str(elmt)
            valstr, unit = txt.split(":")[-1].split()
            val = float(valstr)
            val = energy2hartree(val, unit)

        return val

    @staticmethod
    def _get_ionization(soup) -> Tuple[Optional[float], Optional[float]]:
        # retrieve the ionization energy and the vertical ionization energy
        table_found = False
        for sp in soup.find_all("span"):
            if sp.get_text().strip() == "Ionization Energies (eV)":
                table_found = True
                ie_table = sp.findNext("table")

        if not table_found:
            return None, None

        # convert the string of the entries into energy in hartree
        s2ene = lambda s: energy2hartree(float(s), "eV") if s else None

        ie: Optional[float] = None
        vie: Optional[float] = None
        for i, row in enumerate(ie_table.find_all("tr")):
            if i == 0:
                continue  # skip the first row
            for j, col in enumerate(row.find_all("td")):
                if j == 0:  # ionization energy
                    ie = s2ene(col.get_text().strip())
                elif j == 2:  # vertical ionization energy
                    vie = s2ene(col.get_text().strip())
        return ie, vie

    @staticmethod
    def _get_name(soup) -> str:
        # retrieve the name of the molecule
        title = soup.find(text=re.compile("Experimental data for")).parent
        name = " ".join(title.get_text().split()[3:]).strip()
        return name

    @staticmethod
    def cccbdb_url(casno: str) -> str:
        return "https://cccbdb.nist.gov/exp2x.asp?casno=%s&charge=0" % casno.replace("-", "")

    def __init__(self, casno: str):
        self.casno = casno.strip()
        self.cccbdb_retrieved = False

    def retrieve_cccbdb(self):
        # collect the information from cccbdb page

        # retrieve the page
        for i in range(5):
            try:
                page = requests.get(AtomConf.cccbdb_url(self.casno))
                soup = BeautifulSoup(page.content, "html.parser")

                # retrieve the information from the page
                self.name = AtomConf._get_name(soup)
                print(self.casno, self.name)
                self.atoms, self.all_poss = AtomConf._get_atompos_cccbdb(soup)
                self.enthalpy_0k, self.enthalpy_298k = AtomConf._get_enthalpy(soup)
                self.ie, self.vie = AtomConf._get_ionization(soup)
                self.zpe = AtomConf._get_zpe(soup)
                self.cccbdb_retrieved = True
                break
            except Exception as e:  # RuntimeError, AttributeError:
                print(e)

        return self

    ############### molecule properties ###############
    def s(self) -> str:
        # returns the string representation for DQC
        # position in Bohr unit
        assert self.cccbdb_retrieved

        str_list = []
        for atom, pos in zip(self.atoms, self.all_poss):
            str_list.append("%s %.4f %.4f %.4f" % (atom, *pos))
        return "; ".join(str_list)

    def numel(self) -> int:
        # calculate the number of electrons in the molecule in a neutral state
        assert self.cccbdb_retrieved

        numel = 0
        for atom in self.atoms:
            numel += get_atomz(atom)
        return numel

    def natoms(self) -> int:
        # returns the number of atoms in the molecule
        assert self.cccbdb_retrieved
        return len(self.atoms)

    def atom_counts(self) -> Tuple[List[str], List[int]]:
        # returns the unique list of atoms and its count
        c = Counter(self.atoms)
        return list(c.keys()), list(c.values())

    def ae0(self) -> Optional[float]:
        # calculate the non-relativistic atomization energy from the enthalpy
        # of formation at 0K in hartree
        # ref: eq (1) from https://aip.scitation.org/doi/pdf/10.1063/1.473182
        assert self.cccbdb_retrieved
        if self.enthalpy_0k is None:
            return None
        if self.zpe is None:
            return None

        atom_dHf0 = 0.0
        for atom in self.atoms:
            atom_dHf0 += get_atom_dHf0(atom)
        return atom_dHf0 - self.enthalpy_0k + self.zpe

    ############### database writer ###############
    def ae_db(self, basis: str = "6-311++G**") -> Optional[Dict]:
        # return the database entry for atomization energy,
        # or None if no data is available from CCCBDB
        assert self.cccbdb_retrieved
        ae0 = self.ae0()
        if ae0 is None:
            return None

        res = {}
        res["name"] = "Atomization energy of %s" % self.name
        res["type"] = "ae"
        res["true_val"] = self.ae0()
        res["ref"] = AtomConf.cccbdb_url(self.casno)

        # construct the command
        atoms, counts = self.atom_counts()
        cmd_list = []
        for i, count in enumerate(counts):
            if count == 0:  # should not happen, but just in case
                continue
            elif count == 1:
                cmd_list.append("energy(systems[%d])" % (i + 1))
            else:
                cmd_list.append("%d * energy(systems[%d])" % (count, i + 1))
        cmd = " + ".join(cmd_list)
        cmd = cmd + " - energy(systems[0])"
        res["cmd"] = cmd

        # construct the systems
        systems = [System.create("mol", self.s(), numel0=self.numel(), basis=basis)]
        for atom in atoms:
            systems.append(System.create("mol", "%s 0 0 0" % atom, numel0=get_atomz(atom), basis=basis))
        res["systems"] = systems
        return res

    def vie_db(self, basis: str = "6-311++G**") -> Optional[Dict]:
        # return the database entry for vertical ionization energy,
        # or None if no data is available from CCCBDB
        assert self.cccbdb_retrieved
        if self.vie is None:
            return None

        res = {}
        res["name"] = "Vertical ionization energy of %s" % self.name
        res["type"] = "ie"
        res["true_val"] = self.vie
        res["ref"] = AtomConf.cccbdb_url(self.casno)

        # construct the command
        res["cmd"] = "energy(systems[1]) - energy(systems[0])"

        # construct the systems
        res["systems"] = [
            System.create("mol", self.s(), numel0=self.numel(), basis=basis),
            System.create("mol", self.s(), numel0=self.numel(), basis=basis, charge=1),
        ]
        return res


class System(object):
    caches = {}

    @classmethod
    def create(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls.caches:
            obj = System.init(*args, **kwargs)
            cls.caches[key] = obj
            return obj
        else:
            obj = copy.deepcopy(cls.caches[key])
            return obj

    @classmethod
    def init(cls, tpe: str, moldesc: str, numel0: int, basis: str, charge: int = 0):
        kwargs = {}
        kwargs["moldesc"] = moldesc
        kwargs["basis"] = basis
        if charge != 0:
            kwargs["charge"] = charge
        spin = cls._get_spin(tpe, moldesc, numel0, basis, charge)
        if spin != 0:
            kwargs["spin"] = spin

        res = {}
        res["type"] = tpe
        res["kwargs"] = kwargs
        return res

    @classmethod
    def _get_spin(cls, tpe: str, moldesc: str, numel0: int, basis: str, charge: int) -> int:
        # determine the spin by simulating which spin gives the minimal energy

        assert tpe == "mol"  # only mol for now

        numel = numel0 - charge
        if numel % 2 == 0:
            spins = [0, 2, 4]
        else:
            spins = [1, 3, 5]

        best_ene = 9e99
        for spin in spins:
            try:
                mol = gto.M(atom=moldesc, charge=charge, basis=basis, spin=spin, unit="Bohr")
                if spin == 0:
                    m = dft.RKS(mol)
                else:
                    m = dft.UKS(mol)
                ene = m.kernel()

                if ene < best_ene:
                    best_spin = spin
                    best_ene = ene
            except Exception as e:
                print(e)
        return best_spin

if __name__ == "__main__":
    dbfunc = lambda obj: obj.vie_db()  # for vertical ionization energy entries
    # dbfunc = lambda obj: obj.ae_db()  # for atomization energy entries

    obj = AtomConf("2781-85-3").retrieve_cccbdb()
    entry = dbfunc(obj)
    print(yaml.dump([entry], sort_keys=False))

    # casno_file = "gauss2-casno.txt"
    # write_to = "vie_gauss2.yaml"
    # start_from = 1
    #
    # casnos = []
    # with open(casno_file, "r") as f:
    #     for line in f:
    #         if not line or line.startswith("#"):
    #             continue
    #         casnos.append(line.split(":")[1])
    #
    # skipped = []
    # with open(write_to, "a") as f:
    #     for i, casno in enumerate(casnos):
    #         if i + 1 < start_from:
    #             continue
    #         print(i + 1)
    #         obj = AtomConf(casno).retrieve_cccbdb()
    #         if not obj.cccbdb_retrieved:
    #             skipped.append(casno)
    #         else:
    #             dbobj = dbfunc(obj)
    #             if dbobj is not None:
    #                 f.write(yaml.dump([dbobj], sort_keys=False))
    #                 f.flush()
    #
    # print(skipped)

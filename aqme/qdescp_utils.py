######################################################.
#        This file stores QDESCP functions           #
######################################################.

import json
import sys
import os
import numpy as np
import pandas as pd
import ast
import math
import rdkit
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')
import morfeus
from morfeus import SASA, read_xyz

GAS_CONSTANT = 8.3144621  # J / K / mol
J_TO_AU = 4.184 * 627.509541 * 1000.0  # UNIT CONVERSION
T = 298.15
Hartree = 27.2114 #eV UNIT CONVERSION from 1 hatree = 27.2114 eV

def get_chemical_shifts(json_data, nmr_atoms, nmr_slope, nmr_intercept):
    """
    Retrieves and scales NMR shifts from json files
    """

    if not isinstance(nmr_atoms, list):
        nmr_atoms = ast.literal_eval(nmr_atoms)
    if not isinstance(nmr_slope, list):
        nmr_slope = ast.literal_eval(nmr_slope)
    if not isinstance(nmr_intercept, list):
        nmr_intercept = ast.literal_eval(nmr_intercept)

    atoms = json_data["atoms"]["elements"]["number"]
    tensor = json_data["properties"]["NMR"]["NMR isotopic tensors"]
    shifts = {}
    i = 0
    for atom, ten in zip(atoms, tensor):
        if atom in nmr_atoms:
            # assigning values from arrays
            index = nmr_atoms.index(atom)
            slope_nuc = nmr_slope[index]
            intercept_nuc = nmr_intercept[index]

            scaled_nmr = (intercept_nuc - ten) / (-slope_nuc)
            shifts[i] = scaled_nmr
        else:
            pass
        i += 1

    return shifts

def get_boltz(energy):
    """
    Calculates the Boltzmann weights for a list of energies
    """

    energ = [number - min(energy) for number in energy]

    boltz_sum = 0.0
    for e in energ:
        boltz_sum += math.exp(-e * J_TO_AU / GAS_CONSTANT / T)
    
    weights = []
    for e in energ:
        weight = math.exp(-e * J_TO_AU / GAS_CONSTANT / T) / boltz_sum
        weights.append(weight)

    return weights

def get_boltz_props(json_files, name, boltz_dir, calc_type, self, mol_props, atom_props, smarts_targets, 
                    denovo_mols, denovo_atoms,interpret_mols, interpret_atoms, nmr_atoms=None, nmr_slope=None, nmr_intercept=None, 
                    nmr_experim=None, mol=None, xyz_file=None):
    """
    Retrieves the properties from json files and gives Boltzmann averaged properties for rdkit, NMR and morfues descriptors.
    """
    
    def process_nmr_experim(exp_data, json_data, k):
        """Process NMR experimental data."""
        list_shift = json_data["properties"]["NMR"]["NMR Chemical Shifts"]
        df = pd.DataFrame(list_shift.items(), columns=["atom_idx", f"conf_{k + 1}"])
        df["atom_idx"] += 1
        return exp_data.merge(df, on=["atom_idx"])
    
    def average_properties(boltz, prop_list, smarts_targets, is_atom_prop=True):
        """Calculate average properties based on Boltzmann weights."""
        return average_prop_atom(boltz, prop_list) if is_atom_prop else average_prop_mol(boltz, prop_list)

    def update_avg_json_data(json_data, avg_json_data, prop, avg_prop, smarts_targets):
        """Update avg_json_data with averaged properties."""
        if len(smarts_targets) > 0 or np.isnan(avg_prop).any():
            avg_json_data[prop] = avg_prop
        else:
            avg_json_data[prop] = avg_prop.tolist()

    # Handle NMR experimental data
    exp_data = None
    if calc_type.lower() == "nmr" and nmr_experim:
        try:
            exp_data = pd.read_csv(nmr_experim)
        except FileNotFoundError:
            self.args.log.write(f'\nx  The CSV file with experimental NMR shifts specified ({nmr_experim}) was not found!')
            self.args.log.finalize()
            sys.exit()

    # Calculate Boltzmann weights
    energy = []
    for k, json_file in enumerate(json_files):
        json_data = read_json(json_file)
        energy.append(json_data["total energy"] if calc_type.lower() == "xtb" else json_data["optimization"]["scf"]["scf energies"][-1])
        
        if calc_type.lower() == "nmr":
            json_data["properties"]["NMR"]["NMR Chemical Shifts"] = get_chemical_shifts(json_data, nmr_atoms, nmr_slope, nmr_intercept)
            if exp_data is not None:
                exp_data = process_nmr_experim(exp_data, json_data, k)
            with open(json_file, "w") as outfile:
                json.dump(json_data, outfile)

    boltz = get_boltz(energy)
    avg_json_data = {}
    denovo_json_data = {}
    interpret_json_data = {}

    # Get weighted atomic properties
    #From NMR
    for i, prop in enumerate(atom_props):
        prop_list = [read_json(json_file)[prop] for json_file in json_files]
        avg_prop = average_properties(boltz, prop_list, smarts_targets)
        
        if calc_type.lower() == "nmr":
            dictavgprop = {key: avg_prop[j] for j, key in enumerate(json_data["properties"]["NMR"][prop].keys())}
            avg_json_data[prop] = dictavgprop
            if exp_data is not None:
                df = pd.DataFrame(dictavgprop.items(), columns=["atom_idx", "boltz_avg"])
                df["atom_idx"] = df["atom_idx"].astype(int) + 1
                exp_data = exp_data.merge(df, on=["atom_idx"])
                exp_data["error_boltz"] = abs(exp_data["experimental_ppm"] - exp_data["boltz_avg"])
                qdescp_nmr = nmr_experim.replace(".csv", "_predicted.csv")
                exp_data.round(2).to_csv(qdescp_nmr, index=False)
                self.args.log.write(f"o  The {os.path.basename(qdescp_nmr)} file containing Boltzmann weighted NMR shifts was successfully created in {self.args.initial_dir}")
        else:
            update_avg_json_data(json_data, avg_json_data, prop, avg_prop, smarts_targets)

    # Get weighted molecular properties from XTB
    if calc_type.lower() == "xtb":
        for prop in mol_props:
            prop_list = [read_json(json_file)[prop] for json_file in json_files]
            avg_prop = average_properties(boltz, prop_list, smarts_targets, is_atom_prop=False)
            avg_json_data[prop] = avg_prop

        # Get denovo atomic properties
    for i, prop in enumerate(denovo_atoms):
        prop_list = [read_json(json_file)[prop] for json_file in json_files]
        avg_prop = average_properties(boltz, prop_list, smarts_targets)
        update_avg_json_data(json_data, denovo_json_data, prop, avg_prop, smarts_targets)

        # Get denovo molecular properties
    for prop in denovo_mols:
        prop_list = [read_json(json_file)[prop] for json_file in json_files]
        avg_prop = average_properties(boltz, prop_list, smarts_targets, is_atom_prop=False)
        denovo_json_data[prop] = avg_prop

        # Get interpret atomic properties
    for i, prop in enumerate(interpret_atoms):
        prop_list = [read_json(json_file)[prop] for json_file in json_files]
        avg_prop = average_properties(boltz, prop_list, smarts_targets)
        update_avg_json_data(json_data, interpret_json_data, prop, avg_prop, smarts_targets)

        # Get interpret molecular properties
    for prop in interpret_mols:
        prop_list = [read_json(json_file)[prop] for json_file in json_files]
        avg_prop = average_properties(boltz, prop_list, smarts_targets, is_atom_prop=False)
        interpret_json_data[prop] = avg_prop

    # Calculate RDKit descriptors if molecule is provided
    if mol is not None:
        # Calculate all RDKit properties for avg_json_data
        avg_json_data,_,_ = get_rdkit_properties(avg_json_data, mol)
        
        # Calculate selected RDKit properties for denovo_json_data
        _,denovo_rdkit_json_data,_ = get_rdkit_properties({}, mol)
        
        # Merge selected RDKit properties with denovo_json_data
        denovo_json_data.update(denovo_rdkit_json_data)

        # Calculate selected RDKit properties for interpret_json_data
        _,_,interpret_rdkit_json_data = get_rdkit_properties({}, mol)
        
        # Merge selected RDKit properties with interpret_json_data
        interpret_json_data.update(interpret_rdkit_json_data)

    # Save the averaged properties to a file
    final_boltz_file = os.path.join(boltz_dir, f"{name}_full_boltz.json")
    with open(final_boltz_file, "w") as outfile:
        json.dump(avg_json_data, outfile)
    
    # Save the denovo properties to a second file
    final_denovo_file = os.path.join(boltz_dir, f"{name}_denovo_boltz.json")
    with open(final_denovo_file, "w") as outfile:
        json.dump(denovo_json_data, outfile)

    # Save the interpret properties to a second file
    final_interpret_file = os.path.join(boltz_dir, f"{name}_interpret_boltz.json")
    with open(final_interpret_file, "w") as outfile:
        json.dump(interpret_json_data, outfile)


def average_prop_atom(weights, prop):
    """
    Returns Boltzmann averaged atomic properties
    """
    boltz_avg = []
    for i, p in enumerate(prop):
        if p == 'NaN':
            boltz_avg = 'NaN'
            break
        # None for 0
        boltz_avg.append([0 if number is None else number * weights[i] for number in p])
    if boltz_avg == 'NaN':
        boltz_res = 'NaN'
    else:
        boltz_res = np.sum(boltz_avg, 0)
    return boltz_res

def average_prop_mol(weights, prop):
    """
    Returns Boltzmann averaged molecular properties
    """

    boltz_avg = 0.0
    for i, p in enumerate(prop):
        if p == 'NaN':
            boltz_avg = 'NaN'
            break
        boltz_avg += p * weights[i]
    return boltz_avg


def get_rdkit_properties(avg_json_data, mol):
    """
    Calculates RDKit molecular descriptors
    """

    #Level: denovo descriptors
    denovo_json_data = {}
    #Level: interpret descriptors
    interpret_json_data = {}

    try:
        #level: full
        descrs = Descriptors.CalcMolDescriptors(mol)
        for descr in descrs:
            if descrs[descr] != np.nan and str(descrs[descr]).lower() != 'nan':
                avg_json_data[descr] = descrs[descr]
        
        # descriptors for the level_ denovo
        denovo_json_data["MolLogP"] = descrs.get("MolLogP", None)
        # descriptors for the level: interpret
        interpret_json_data["MolLogP"] = descrs.get("MolLogP", None)

    except AttributeError:
        avg_json_data["NHOHCount"] = rdkit.Chem.Lipinski.NHOHCount(mol)
        avg_json_data["FractionCSP3"] = rdkit.Chem.Lipinski.FractionCSP3(mol)
        avg_json_data["NOCount"] = rdkit.Chem.Lipinski.NOCount(mol)
        avg_json_data["NumAliphaticRings"] = rdkit.Chem.Lipinski.NumAliphaticRings(mol)
        avg_json_data["NumAromaticRings"] = rdkit.Chem.Lipinski.NumAromaticRings(mol)
        avg_json_data["NumHAcceptors"] = rdkit.Chem.Lipinski.NumHAcceptors(mol)
        avg_json_data["NumHDonors"] = rdkit.Chem.Lipinski.NumHDonors(mol)
        avg_json_data["NumHeteroatoms"] = rdkit.Chem.Lipinski.NumHeteroatoms(mol)
        avg_json_data["NumRotatableBonds"] = rdkit.Chem.Lipinski.NumRotatableBonds(mol)
        avg_json_data["TPSA"] = rdkit.Chem.Descriptors.TPSA(mol)
        avg_json_data["MolLogP"] = rdkit.Chem.Descriptors.MolLogP(mol)

        # descriptors for the level_ denovo
        denovo_json_data["MolLogP"] = avg_json_data["MolLogP"]
        # descriptors for the level: interpret
        interpret_json_data["MolLogP"] = avg_json_data["MolLogP"]


    return avg_json_data, denovo_json_data, interpret_json_data


def read_gfn1(file):
    """
    Read .gfn1 output file created from xTB. Return data.
    """

    if file.find(".gfn1") > -1:
        f = open(file, "r")
        data = f.readlines()
        f.close()

        for i in range(0, len(data)):
            if data[i].find("Mulliken/CM5 charges") > -1:
                start = i + 1
                break
        for j in range(start, len(data)):
            if (
                data[j].find("Wiberg/Mayer (AO) data") > -1
                or data[j].find("generalized Born model") > -1
            ):
                end = j - 1
                break

        pop_data = data[start:end]
        mulliken, cm5, s_prop, p_prop, d_prop = [], [], [], [], []
        for line in pop_data:
            item = line.split()
            q_mull = round(float(item[-5]),5)
            q_cm5 = round(float(item[-4]),5)
            s_prop_ind = round(float(item[-3]),3)
            p_prop_ind = round(float(item[-2]),3)
            d_prop_ind = round(float(item[-1]),3)
            mulliken.append(q_mull)
            cm5.append(q_cm5)
            s_prop.append(s_prop_ind)
            p_prop.append(p_prop_ind)
            d_prop.append(d_prop_ind)

        localgfn1 = {
        "mulliken charges": mulliken,
        "cm5 charges": cm5,
        "s proportion": s_prop,
        "p proportion": p_prop,
        "d proportion": d_prop,
        }

        return localgfn1

def read_wbo(file):
    """
    Read wbo output file created from xTB. Return data.
    """

    if file.find(".wbo") > -1:
        f = open(file, "r")
        data = f.readlines()
        f.close()

        bonds, wbos = [], []
        for line in data:
            item = line.split()
            bond = [int(item[0]), int(item[1])]
            wbo = round(float(item[2]),3)
            bonds.append(bond)
            wbos.append(wbo)
        return bonds, wbos

def calculate_global_CDFT_descriptors(file):
    """
    Read .gfn1 output file created from xTB and calculate CDFT descriptors with FDA approximations part 1.
    """
    if not file.endswith(".gfn1"):
        raise ValueError("Missing .gfn1 output file")

    try:
        with open(file, "r") as f:
            data = f.readlines()
    except IOError:
        raise IOError("Error opening or reading the file.")
    
    delta_SCC_IP, delta_SCC_EA, electrophilicity_index = None, None, None

    # Extract relevant values from the file
    for line in data:
        if "delta SCC IP (eV):" in line:
            delta_SCC_IP = float(line.split()[-1])
        elif "delta SCC EA (eV):" in line:
            delta_SCC_EA = float(line.split()[-1])
        elif "Global electrophilicity index (eV):" in line:
            electrophilicity_index = float(line.split()[-1])

    # Check if required descriptors were found
    if delta_SCC_IP is None or delta_SCC_EA is None:
        raise ValueError("Could not find delta_SCC_IP and delta_SCC_EA descriptors in the file")

    # Calculate CDFT descriptors
    chemical_hardness = round((delta_SCC_IP - delta_SCC_EA), 4)
    chemical_softness = round(1 / chemical_hardness, 4) if chemical_hardness != 0 else None
    chemical_potential = round(-(delta_SCC_IP + delta_SCC_EA) / 2, 4)
    mulliken_electronegativity = round(-chemical_potential, 4)

    try:
        electrodonating_power_index = round(((delta_SCC_IP + 3 * delta_SCC_EA)**2) / (8 * chemical_hardness), 4)
        electroaccepting_power_index = round(((3 * delta_SCC_IP + delta_SCC_EA)**2) / (8 * chemical_hardness), 4)
        nucleophilicity_index = round(10 / electroaccepting_power_index, 4) if electroaccepting_power_index != 0 else None
    except ZeroDivisionError:
        electrodonating_power_index = None
        electroaccepting_power_index = None
        nucleophilicity_index = None
        print("Warning: Division by zero encountered in power index calculations.")

    electrofugality = round(-delta_SCC_EA + electrophilicity_index, 4) if electrophilicity_index is not None else None
    nucleofugality = round(delta_SCC_IP + electrophilicity_index, 4) if electrophilicity_index is not None else None
    intrinsic_reactivity_index = round((delta_SCC_IP + delta_SCC_EA) / chemical_hardness, 4) if chemical_hardness != 0 else None
    net_electrophilicity = round((electrodonating_power_index - electroaccepting_power_index), 4) if electrodonating_power_index is not None and electroaccepting_power_index is not None else None

    cdft_descriptors = {
        "IP (eV)": delta_SCC_IP,
        "EA (eV)": delta_SCC_EA,
        "Electrophilicity index (eV)": electrophilicity_index,
        "Chemical Hardness (eV)": chemical_hardness,
        "Chemical Softness (1/eV)": chemical_softness,
        "Chemical Potential (eV)": chemical_potential,
        "Mulliken Electronegativity (eV)": mulliken_electronegativity,
        "Electrodonating Power Index (eV)": electrodonating_power_index,
        "Electroaccepting Power Index (eV)": electroaccepting_power_index,
        "Nucleophilicity Index (eV)": nucleophilicity_index,
        "Electrofugality (eV)": electrofugality,
        "Nucleofugality (eV)": nucleofugality,
        "Intrinsic Reactivity Index (eV)": intrinsic_reactivity_index,
        "Net Electrophilicity (eV)": net_electrophilicity
    }
    
    # # Print for control (remove these lines in production)
    # print("Global CDFT part 1")
    # for key, value in cdft_descriptors.items():
    #     print(f"{key}: {value}")

    return cdft_descriptors

def calculate_global_CDFT_descriptors_part2(file, file_Nminus1, file_Nminus2, file_Nplus1, file_Nplus2, cdft_descriptors):
    """
    Read .gfn1 output file created from xTB and calculate CDFT descriptors with FDA approximations part 2
    """
    corr_xtb = 4.8455

    def extract_scc_energy(lines, filename):
        for line in lines:
            if "SCC energy" in line:
                return float(line.split()[3])
        raise ValueError(f"Could not find SCC energy value in the file: {filename}")

    with open(file, "r") as f:
        data = f.readlines()
    with open(file_Nminus1, "r") as f1:
        data1 = f1.readlines()
    with open(file_Nminus2, "r") as f2:
        data2 = f2.readlines()
    with open(file_Nplus1, "r") as f3:
        data3 = f3.readlines()
    with open(file_Nplus2, "r") as f4:
        data4 = f4.readlines()

    scc_energy = extract_scc_energy(data, file) * Hartree
    scc_energy_Nminus1 = extract_scc_energy(data1, file_Nminus1) * Hartree
    scc_energy_Nminus2 = extract_scc_energy(data2, file_Nminus2) * Hartree
    scc_energy_Nplus1 = extract_scc_energy(data3, file_Nplus1) * Hartree
    scc_energy_Nplus2 = extract_scc_energy(data4, file_Nplus2) * Hartree

    delta_SCC_IP = cdft_descriptors.get("IP (eV)")
    delta_SCC_EA = cdft_descriptors.get("EA (eV)")
    chemical_hardness = cdft_descriptors.get("Chemical Hardness (eV)")

    Vertical_second_IP = None
    Vertical_second_EA = None
    hyper_hardness = None
    Global_hypersoftness = None
    Electrophilic_descriptor = None
    w_cubic = None
    
    if scc_energy is not None and scc_energy_Nminus1 is not None and scc_energy_Nminus2 is not None and scc_energy_Nplus1 is not None and scc_energy_Nplus2 and delta_SCC_IP is not None and delta_SCC_EA is not None:
        Vertical_second_IP = round((((scc_energy_Nminus2 - scc_energy_Nminus1) - corr_xtb)), 4)
        Vertical_second_EA = round((((scc_energy_Nplus1 - scc_energy_Nplus2) + corr_xtb)), 4)
        
        # hyper_hardness
        hyper_hardness = round((-((0.5) * (delta_SCC_IP + delta_SCC_EA - Vertical_second_IP - Vertical_second_EA))), 4)

        # Global_hypersoftness
        Global_hypersoftness = round((hyper_hardness / ((chemical_hardness) ** 3)), 4)
        
        # Electrophilic descriptor
        try:
            A = ((scc_energy_Nplus1 - scc_energy) + corr_xtb)
            c = (Vertical_second_IP - (2 * delta_SCC_IP) + A) / ((2 * Vertical_second_IP) - delta_SCC_IP - A)
            a = -((delta_SCC_IP + A) / 2) + (((delta_SCC_IP - A) / 2) * c)
            b = ((delta_SCC_IP - A) / 2) - (((delta_SCC_IP + A) / 2) * c)
            Gamma = (-3 * c) * (b - (a * c))
            Eta = 2 * (b - (a * c))
            chi = -a
            Mu = a
            
            # Checking the square root
            discriminant = Eta ** 2 - (2 * Gamma * Mu)
            if discriminant < 0:
                raise ValueError(f"Negative discriminant: cannot compute the square root of {discriminant}. Electrophilic descriptor was not calculated")
            
            inter_phi = math.sqrt(discriminant)
            Phi = inter_phi - Eta
            Electrophilic_descriptor = round(((chi * (Phi / Gamma)) - (((Phi / Gamma) ** 2) * ((Eta / 2) + (Phi / 6)))), 4)

        except ValueError as e:
            Phi = None
            Electrophilic_descriptor = None
        
        # w cubic electrophilicity index
        try:
            Gamma_cubic = 2 * delta_SCC_IP - Vertical_second_IP - delta_SCC_EA
            Eta_cubic = delta_SCC_IP - delta_SCC_EA
            
            # Check if Eta_cubic != 0
            if Eta_cubic == 0:
                raise ZeroDivisionError("Eta_cubic is zero, which would cause a division by zero in the calculation.")
            
            Mu_cubic = (1 / 6) * ((-2 * delta_SCC_EA) - (5 * delta_SCC_IP) + Vertical_second_IP)
            w_cubic = round(((Mu_cubic ** 2) / (2 * Eta_cubic)) * (1 + ((Mu_cubic / (3 * (Eta_cubic) ** 2)) * Gamma_cubic)), 4)

        except ZeroDivisionError as e:
            w_cubic = None 
        except TypeError as e:
            w_cubic = None  
    
    cdft_descriptors2 = {
        "Vertical second IP (eV)": Vertical_second_IP,
        "Vertical second EA (eV)": Vertical_second_EA,
        "Hyper Hardness (eV)": hyper_hardness,
        "Global Hypersoftness (1/eV^2)": Global_hypersoftness,
        "Electrophilic descriptor (eV)": Electrophilic_descriptor,
        "W cubic electrophilicity index (eV)": w_cubic
    }

    # # Print for control (remove these lines in production)
    # print("Global CDFT part 2")
    # for key, value in cdft_descriptors2.items():
    #     print(f"{key}: {value}")

    return cdft_descriptors2

def calculate_local_CDFT_descriptors(file_fukui, cdft_descriptors, cdft_descriptors2):
    """
    Read fukui output file created from XTB option. Return data.
    """
    if not file_fukui.endswith(".fukui"):
        raise ValueError("Missing .fukui output file")

    with open(file_fukui, "r") as f:
        data = f.readlines()

    # Searching Fukuis
    f_pos, f_negs, f_rads = [], [], []
    start, end = None, None

    # Search for the start and end lines in the file_fukui
    for i, line in enumerate(data):
        if "f(+) " in line:
            start = i + 1
        elif "-------------" in line and start is not None:
            end = i
            break

    # Ensure start and end indices were found
    if start is not None and end is not None:
        fukui_data = data[start:end]
        for line in fukui_data:
            try:
                f_po, f_neg, f_rad = map(lambda x: round(float(x), 4), line.split()[-3:])
                f_pos.append(f_po)
                f_negs.append(f_neg)
                f_rads.append(f_rad)
            except ValueError:
                continue
    else:
        print("Fukui data not found in the file, please check the '.fukui' file.")

    # Check if the lists are populated
    if not f_pos or not f_negs or not f_rads:
        print("Fukui data not found in the file, please check the '.fukui' file.")
        return
    
    # Extract necessary values from the provided dictionaries
    chemical_softness = cdft_descriptors.get("Chemical Softness (1/eV)")
    Global_hypersoftness = cdft_descriptors2.get("Global Hypersoftness (1/eV^2)")
    electrophilicity_index = cdft_descriptors.get("Electrophilicity index (eV)")
    nucleophilicity_index = cdft_descriptors.get("Nucleophilicity Index (eV)")

    # Calculating local Descriptors part 2
    dual_descriptor = [round(f_po - f_neg, 4) for f_po, f_neg in zip(f_pos, f_negs)]

    s_pos = [round(chemical_softness * f_po, 4) for f_po in f_pos]
    s_negs = [round(chemical_softness * f_neg, 4) for f_neg in f_negs]
    s_rads = [round(chemical_softness * f_rad, 4) for f_rad in f_rads]

    Relative_nucleophilicity = [
        round(s_neg / s_po, 4) if s_po != 0 else None for s_neg, s_po in zip(s_negs, s_pos)
    ]
    Relative_electrophilicity = [
        round(s_po / s_neg, 4) if s_neg != 0 else None for s_neg, s_po in zip(s_negs, s_pos)
    ]

    Grand_canonical_dual_descriptor = [
        round(Global_hypersoftness * dual, 4) for dual in dual_descriptor
    ]

    w_pos = [round(electrophilicity_index * f_po, 4) for f_po in f_pos]
    w_negs = [round(electrophilicity_index * f_neg, 4) for f_negs in f_negs]
    w_rads = [round(electrophilicity_index * f_rad, 4) for f_rad in f_rads]

    Multiphilic_descriptor = [
        round(electrophilicity_index * dual, 4) for dual in dual_descriptor
    ]

    Nu_pos = [round(nucleophilicity_index * f_po, 4) for f_po in f_pos]
    Nu_negs = [round(nucleophilicity_index * f_neg, 4) for f_neg in f_negs]
    Nu_rads = [round(nucleophilicity_index * f_rad, 4) for f_rad in f_rads]

    localDescriptors = {
        "F+": f_pos,
        "F-": f_negs,
        "F0": f_rads,
        "dual_descriptor": dual_descriptor,
        "s+": s_pos,
        "s-": s_negs,
        "srad": s_rads,
        "s+/s-": Relative_nucleophilicity,
        "s-/s+": Relative_electrophilicity,
        "Grand_Canonical_Dual_Descriptor": Grand_canonical_dual_descriptor,
        "w+": w_pos,
        "w-": w_negs,
        "wrad": w_rads,
        "Multiphilic_descriptor": Multiphilic_descriptor,
        "Nu+": Nu_pos,
        "Nu-": Nu_negs,
        "Nurad": Nu_rads
    }

    # # Print for control
    # print("Local Descriptors")
    # for key, value in localDescriptors.items():
    #     print(f"{key}: {value}")

    return localDescriptors

def read_xtb(file):
    """
    Read xtb.out file and return a dictionary of extracted properties.
    """
    try:
        with open(file, "r") as f:
            data = f.readlines()
    except IOError:
        raise IOError(f"Error opening or reading the file: {file}")

    # Initialize variables
    energy, homo_lumo, homo, lumo = np.nan, np.nan, np.nan, np.nan
    dipole_module, Fermi_level, transition_dipole_moment = np.nan, np.nan, np.nan
    total_charge, total_SASA = np.nan, np.nan
    total_C6AA, total_C8AA, total_alpha = np.nan, np.nan, np.nan
    atoms, numbers, chrgs = [], [], []
    covCN, C6AA, alpha = [], [], []
    born_rad, SASA, h_bond = [], [], []

    # Parsing file data
    for i, line in enumerate(data):
        if "SUMMARY" in line:
            energy = float(data[i + 2].split()[3])
        elif "total charge" in line:
            total_charge = int(float(data[i].split()[3]))
        elif "(HOMO)" in line:
            if data[i].split()[3] != "(HOMO)":
                homo = round(float(data[i].split()[3]), 4)
                homo_occ = round(float(data[i].split()[1]), 4)
            else:
                homo = round(float(data[i].split()[2]), 4)
                homo_occ = 0
        elif "(LUMO)" in line:
            if data[i].split()[3] != "(LUMO)":
                lumo = round(float(data[i].split()[3]), 4)
                lumo_occ = round(float(data[i].split()[1]), 4)
            else:
                lumo = round(float(data[i].split()[2]), 4)
                lumo_occ = 0
        elif "molecular dipole:" in line:
            dipole_module = float(data[i + 3].split()[-1])
        elif "transition dipole moment" in line:
            transition_dipole_moment = float(data[i + 2].split()[-1])
        elif "Fermi-level" in line:
            Fermi_level = float(data[i].split()[-2])

    homo_lumo = round(float(lumo - homo), 4)

    # Getting atomic properties related to charges, dispersion, etc.
    start, end = 0, 0
    for j in range(len(data)):
        if "#   Z          covCN" in data[j]:
            start = j + 1
            break
    for k in range(start, len(data)):
        if "Mol. " in data[k]:
            end = k - 1
            total_C6AA = float(data[k].split()[-1])
            total_C8AA = float(data[k + 1].split()[-1])
            total_alpha = float(data[k + 2].split()[-1])
            break

    chrg_data = data[start:end]
    for line in chrg_data:
        item = line.split()
        numbers.append(int(item[0]))
        atoms.append(item[2])
        covCN.append(float(item[3]))
        chrgs.append(float(item[4]))
        C6AA.append(float(item[5]))
        alpha.append(float(item[6]))

    # Getting atomic properties related to solvent
    start_solv, end_solv = 0, 0
    for j in range(len(data)):
        if "#   Z     Born rad" in data[j]:
            start_solv = j + 1
            break
    for k in range(start_solv, len(data)):
        if "total SASA " in data[k]:
            end_solv = k - 1
            total_SASA = float(data[k].split()[-1])
            break

    solv_data = data[start_solv:end_solv]
    for line in solv_data:
        item = line.split()
        born_rad.append(float(item[3]))
        SASA.append(float(item[4]))
        try:
            h_bond.append(float(item[5]))
        except IndexError:
            h_bond.append(0.0)

    properties_dict = {
        "total energy": energy,
        "Total charge": total_charge,
        "HOMO-LUMO gap": homo_lumo,
        "HOMO": homo,
        "LUMO": lumo,
        "atoms": atoms,
        "numbers": numbers,
        "charges": chrgs, 
        "Dipole module": dipole_module,
        "Fermi-level": Fermi_level,
        "transition_dipole_moment": transition_dipole_moment,
        "Coordination numbers": covCN,
        "Dispersion coefficient C6": C6AA,
        "Polarizability alpha": alpha,
        "HOMO occupancy": homo_occ,
        "LUMO occupancy": lumo_occ,
        "Born radii": born_rad,
        "Atomic SASAs": SASA,
        "Solvent H bonds": h_bond,
        "Total SASA": total_SASA,
        "Total dispersion C6": total_C6AA,
        "Total dispersion C8": total_C8AA,
        "Total polarizability alpha": total_alpha,
    }

    return properties_dict


def read_fod(file):
    """
    Read xtb.fod files. Return FOD-related properties.
    """

    f = open(file, "r")
    data = f.readlines()
    f.close()

    # get fractional occupation density (FOD)
    for j in range(0, len(data)):
        if data[j].find("Loewdin FODpop") > -1:
            start_fod = j + 1
            total_fod = float(data[j - 2].split()[-1])
            break
    for k in range(start_fod, len(data)):
        if data[k].find("Wiberg/Mayer") > -1:
            end_fod = k - 1
            break

    fod_data = data[start_fod:end_fod]
    fod, s_prop_fod, p_prop_fod, d_prop_fod = [], [], [], []
    for line in fod_data:
        item = line.split()
        fod.append(float(item[1]))
        s_prop_fod.append(float(item[2]))
        p_prop_fod.append(float(item[3]))
        d_prop_fod.append(float(item[4]))

    properties_FOD = {
    "Total FOD": total_fod,
    "FOD": fod,
    "FOD s proportion": s_prop_fod,
    "FOD p proportion": p_prop_fod,
    "FOD d proportion": d_prop_fod,
}

    return properties_FOD

def read_json(file):
    """
    Takes json files and parses data into pandas table. Returns data.
    """

    if file.find(".json") > -1:
        f = open(file, "r")  # Opening JSON file
        data = json.loads(f.read())  # read file
        f.close()
        return data
    else:
        pass
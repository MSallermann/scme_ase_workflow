from snakemake.script import snakemake

from enum import Enum
from ase.io import read, write, Trajectory
from ase.units import Bohr, fs, Hartree
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ase.optimize import BFGS, BFGSLineSearch, FIRE2
import json
import numpy as np
import time

from pathlib import Path
from typing import Optional

import pyscme
from pyscme.parameters import parameter_H2O
from pyscme.scme_calculator import SCMECalculator

from pydantic import BaseModel, ConfigDict


para_dict = {
    "te": 1.1045 / Bohr,
    "td": 7.5548 * Bohr,
    "Ar": 8149.63 / Hartree,
    "Br": -0.5515,
    "Cr": -3.4695 * Bohr,
    "r_Br": 1.0 / Bohr,
    "rc_Disp": 8.0 / Bohr,
    "rc_Core": 7.5 / Bohr,
    "rc_Elec": 9.0 / Bohr,
    "w_rc_Elec": 2.0 / Bohr,
    "w_rc_Disp": 2.0 / Bohr,
    "w_rc_Core": 2.0 / Bohr,
    "C6": 46.4430e0,
    "C8": 1141.7000e0,
    "C10": 33441.0000e0,
    "scf_convcrit": 1e-6,
    "scf_policy": pyscme.SCFPolicy.strict,
}


class Method(Enum):
    BFGS = "BFGS"
    VelocityVerlet = "VelocityVerlet"
    Langevin = "Langevin"
    NoseHoover = "NoseHoover"
    Fire = "Fire"


class ASERunParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Method
    interval_properties: int = 1000
    n_iter: Optional[int] = None
    timestep: float = 1
    temperature: Optional[float] = None
    fmax: Optional[float] = 0.05
    pbc: list[bool] = [True, True, False]
    trajectory_interval: int = 1
    constrain_water: bool = True
    langevin_friction: float = 0.01 / fs
    nose_hoover_damping_factor: float = (
        100  # noose hoover damping factor as a multiple of the timestep
    )


def write_data_to_json(atoms, path: Path, additional_data=None):
    with open(path, "w") as f:
        res_dict = dict(
            energy_core=atoms.calc.energy_core,
            energy_dispersion=atoms.calc.energy_dispersion,
            energy_electrostatic=atoms.calc.energy_electrostatic,
            energy_monomer=atoms.calc.energy_monomer,
            energy_pot=atoms.calc.energy,
            energy_kin=atoms.get_kinetic_energy(),
            energy_tot=atoms.get_total_energy(),
            dipole=atoms.calc.results["dipole"].tolist(),
            quadrupole=np.sum(atoms.calc.scme.quadrupole_moments, axis=0).tolist(),
            box_length=list(atoms.cell.cellpar()[:3]),
            box_volume=np.prod(atoms.cell.cellpar()[:3]),
            n_water=int(len(atoms) / 3),
        )

        if not additional_data is None:
            res_dict.update(additional_data)

        json.dump(res_dict, f, indent=4)


def constrain_water(atoms):
    import pyfixi
    from pyfixi.constraints import FixBondLengths

    n_atoms = len(atoms)
    n_molecules = int(n_atoms / 3)

    pairs = []
    for i_molecule in range(n_molecules):
        iO = 3 * i_molecule
        iH1 = iO + 1
        iH2 = iO + 2

        pairs.append([iO, iH1])
        pairs.append([iO, iH2])
        pairs.append([iH1, iH2])

    atoms.set_constraint(FixBondLengths(pairs, tolerance=1e-6))


def construct_calculator(atoms, para_dict):
    return SCMECalculator(atoms=atoms, **para_dict)


def main(
    input_xyz: Path,
    ase_params: ASERunParams,
    logfile: Optional[Path] = None,
    properties_file: Optional[Path] = None,
    output_xyz: Optional[Path] = None,
    trajectory_file: Optional[Path] = None,
    initial_data: Optional[Path] = None,
    final_data: Optional[Path] = None,
):

    # Read the system using ASE
    with open(input_xyz, "r") as f:
        atoms = read(f, format="extxyz")

    atoms.calc = construct_calculator(atoms, para_dict)
    atoms.set_pbc(ase_params.pbc)
    parameter_H2O.Assign_parameters_H20(atoms.calc.scme)

    if ase_params.constrain_water:
        constrain_water(atoms)

    dt = ase_params.timestep * fs

    if ase_params.method == Method.VelocityVerlet:
        dyn = VelocityVerlet(
            atoms,
            timestep=dt,
            logfile=logfile,
        )
    elif ase_params.method == Method.BFGS:
        dyn = BFGS(atoms, logfile=logfile)
    elif ase_params.method == Method.Fire:
        dyn = FIRE2(atoms, logfile=logfile)
    elif ase_params.method == Method.Langevin:
        MaxwellBoltzmannDistribution(atoms, temperature_K=ase_params.temperature)
        dyn = Langevin(
            atoms,
            timestep=dt,
            temperature_K=ase_params.temperature,
            friction=ase_params.langevin_friction,
            logfile=logfile,
        )
    elif ase_params.method == Method.NoseHoover:
        MaxwellBoltzmannDistribution(atoms, temperature_K=ase_params.temperature)
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=dt,
            temperature_K=ase_params.temperature,
            tdamp=ase_params.nose_hoover_damping_factor * dt,
            logfile=logfile,
        )

    if not initial_data is None:
        atoms.calc.calculate(atoms)
        write_data_to_json(atoms, initial_data)

    if not trajectory_file is None:
        trajectory_obj = Trajectory(trajectory_file, mode="w", atoms=atoms)
        dyn.attach(trajectory_obj, interval=ase_params.trajectory_interval)

    if not properties_file is None:
        from ase_extras.property_writer import PropertyWriter

        writer = PropertyWriter(
            atoms=atoms,
            file=properties_file,
            dyn=dyn,
            properties=["total_energy", "temperature", "potential_energy"],
        )
        dyn.attach(writer.log_to_file, interval=ase_params.interval_properties)

    t_start = time.time()
    if ase_params.method in [Method.BFGS, Method.Fire]:
        dyn.run(steps=ase_params.n_iter, fmax=ase_params.fmax)
    else:
        dyn.run(steps=ase_params.n_iter)
    t_end = time.time()
    elapsed_time = t_end - t_start

    if not final_data is None:
        write_data_to_json(
            atoms, final_data, additional_data=dict(time_seconds=elapsed_time)
        )

    dyn.close()

    if not output_xyz is None:
        with open(output_xyz, "w") as f:
            write(f, atoms)


if __name__ == "__main__":

    ase_params = ASERunParams(**snakemake.params["ase_params"])

    scme_params = snakemake.params.get("scme_params", None)

    if not scme_params is None:
        para_dict.update(scme_params)

    input_xyz = Path(snakemake.input["xyz_file"])

    pyscme.set_num_threads(snakemake.resources.cpus_per_task)
    pyfixi.set_num_threads(snakemake.resources.cpus_per_task)

    main(
        input_xyz=input_xyz,
        ase_params=ase_params,
        logfile=snakemake.output.get("logfile"),
        properties_file=snakemake.output.get("properties_file"),
        output_xyz=snakemake.output.get("xyz_file"),
        trajectory_file=snakemake.output.get("trajectory_file"),
        initial_data=snakemake.output.get("initial_data"),
        final_data=snakemake.output.get("final_data"),
    )

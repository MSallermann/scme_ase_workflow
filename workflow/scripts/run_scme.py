from snakemake.script import snakemake

from enum import Enum

from ase import Atoms
from ase.io import read, write, Trajectory
from ase.units import fs
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS, FIRE2
import json
import numpy as np
import time
from ase.units import Bohr, Hartree

from pathlib import Path
from typing import Optional

import pyfixi
from pyfixi.constraints import FixBondLengths

import pyscme
from pyscme.parameters import parameter_H2O
from pyscme.scme_calculator import SCMECalculator

from pydantic import BaseModel, ConfigDict


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
    box_lengths: Optional[list[float]] = None
    scale: Optional[float] = None


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

        if additional_data is not None:
            res_dict.update(additional_data)

        json.dump(res_dict, f, indent=4)


def constrain_water(atoms):
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


def scale_atoms(atoms: Atoms, scale: float):
    cell_old = atoms.get_cell()
    cell_new = cell_old * scale
    atoms.set_cell(cell_new, scale_atoms=True)
    return atoms


def main(
    input_xyz: Path,
    ase_params: ASERunParams,
    scme_params: dict,
    logfile: Optional[Path] = None,
    properties_file: Optional[Path] = None,
    output_xyz: Optional[Path] = None,
    trajectory_file: Optional[Path] = None,
    initial_data: Optional[Path] = None,
    final_data: Optional[Path] = None,
    initial_dipoles: Optional[Path] = None,
    initial_quadrupoles: Optional[Path] = None,
    final_dipoles: Optional[Path] = None,
    final_quadrupoles: Optional[Path] = None,
):
    # Read the system using ASE
    with open(input_xyz, "r") as f:
        atoms = read(f, format="extxyz")

    if ase_params.box_lengths is not None:
        atoms.set_cell(ase_params.box_lengths, scale_atoms=False)

    if ase_params.constrain_water:
        constrain_water(atoms)

    if ase_params.scale is not None:
        scale_atoms(atoms, ase_params.scale)
        atoms.set_velocities(np.zeros((len(atoms), 3)))

    atoms.calc = construct_calculator(atoms, scme_params)
    atoms.set_pbc(ase_params.pbc)
    parameter_H2O.Assign_parameters_H20(atoms.calc.scme)

    dt = ase_params.timestep * fs

    if ase_params.method == Method.VelocityVerlet:
        dyn = VelocityVerlet(
            atoms,
            timestep=dt,
            logfile=logfile,
        )
    elif ase_params.method == Method.BFGS:
        atoms.set_velocities(np.zeros((len(atoms), 3)))
        dyn = BFGS(atoms, logfile=logfile)
    elif ase_params.method == Method.Fire:
        atoms.set_velocities(np.zeros((len(atoms), 3)))
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

    atoms.calc.calculate(atoms)

    if initial_data is not None:
        write_data_to_json(atoms, initial_data)

    if initial_dipoles is not None:
        np.save(initial_dipoles, atoms.calc.scme.dipole_moments)

    if initial_quadrupoles is not None:
        np.save(initial_quadrupoles, atoms.calc.scme.quadrupole_moments)

    if trajectory_file is not None:
        trajectory_obj = Trajectory(trajectory_file, mode="w", atoms=atoms)
        dyn.attach(trajectory_obj, interval=ase_params.trajectory_interval)

    if properties_file is not None:
        from ase_extras.property_writer import PropertyWriter

        writer = PropertyWriter(
            atoms=atoms,
            file=properties_file,
            dyn=dyn,
            properties=["nsteps", "total_energy", "temperature", "potential_energy"],
        )
        dyn.attach(writer.log_to_file, interval=ase_params.interval_properties)

    t_start = time.time()
    if ase_params.method in [Method.BFGS, Method.Fire]:
        dyn.run(steps=ase_params.n_iter, fmax=ase_params.fmax)
    else:
        dyn.run(steps=ase_params.n_iter)
    t_end = time.time()
    elapsed_time = t_end - t_start

    if final_data is not None:
        write_data_to_json(
            atoms, final_data, additional_data=dict(time_seconds=elapsed_time)
        )

    if final_dipoles is not None:
        np.save(final_dipoles, atoms.calc.scme.dipole_moments)

    if final_quadrupoles is not None:
        np.save(final_quadrupoles, atoms.calc.scme.quadrupole_moments)

    dyn.close()

    if output_xyz is not None:
        with open(output_xyz, "w") as f:
            write(f, atoms)


if __name__ == "__main__":
    ase_params = ASERunParams(**snakemake.params["ase_params"])

    default_scme_params = {
        "dispersion": {
            "td": 7.5548 * Bohr,
            "rc": 8.0 / Bohr,
            "C6_OO": 46.4430e0,
            "C8_OO": 1141.7000e0,
            "C10_OO": 33441.0000e0,
        },
        "repulsion": {
            "Ar_OO": 8149.63 / Hartree,
            "Br_OO": -0.5515,
            "Cr_OO": -3.4695 * Bohr,
            "r_Br": 1.0 / Bohr,
            "rc": 7.5 / Bohr,
        },
        "electrostatic": {
            "scf_convcrit": 1e-8,
            "NC": [1, 2, 1],
            "scf_policy": pyscme.SCFPolicy.strict,
            "te": 1.2 / Bohr,
            "max_iter_scf": 500,
            "rc": 9.0 / Bohr,
        },
        "dms": False,
        "qms": False,
    }

    scme_params = snakemake.params.get("scme_params", None)
    scme_params = default_scme_params.update(scme_params)

    input_xyz = Path(snakemake.input["xyz_file"])

    pyscme.set_num_threads(snakemake.resources.cpus_per_task)
    pyfixi.set_num_threads(snakemake.resources.cpus_per_task)

    main(
        input_xyz=input_xyz,
        ase_params=ase_params,
        scme_params=scme_params,
        logfile=snakemake.output.get("logfile"),
        properties_file=snakemake.output.get("properties_file"),
        output_xyz=snakemake.output.get("xyz_file"),
        trajectory_file=snakemake.output.get("trajectory_file"),
        initial_data=snakemake.output.get("initial_data"),
        final_data=snakemake.output.get("final_data"),
        initial_dipoles=snakemake.output.get("initial_dipoles"),
        initial_quadrupoles=snakemake.output.get("initial_quadrupoles"),
        final_dipoles=snakemake.output.get("final_dipoles"),
        final_quadrupoles=snakemake.output.get("final_quadrupoles"),
    )

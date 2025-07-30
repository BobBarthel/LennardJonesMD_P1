#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LJ_gas_run_MD.py

Main program for running molecular dynamics simulations using Lennard-Jones particles.
Initializes the system, runs the integrator loop, records energy and trajectory data, 
and visualizes results.

Author: Bettina Keller
Created: May 28, 2025

This script imports all classes and functions from md_simulation.py and controls
the simulation workflow.

"""

#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------
import os
import nanoid as nano
import numpy as np
from scipy.constants import R
import matplotlib.pyplot as plt
import pandas as pd

import time
from datetime import datetime

from LJ_gas import(
    ParticleSystem,
    SimulationParameters,
    simulate_NVE_step,
    simulate_NVT_step,
    initialize_positions,
    initialize_velocities,
    get_neighbors,
    calculate_force,
    density,
    write_xyz_trajectory,
    potential_energy,
    kinetic_energy,
    instantaneous_temperature,
    ideal_gas_pressure
    )

#----------------------------------------------------------------
#   F U N C T I O N S
#----------------------------------------------------------------
# Define tic and toc functions
def tic():
    """Start a timer."""
    global _tic_time
    _tic_time = time.time()

def toc():
    """Stop the timer and return the elapsed time in seconds."""

    elapsed_time = None
    
    if '_tic_time' in globals():
        elapsed_time = time.time() - _tic_time
    
    else:
        print("Error: tic() was not called before toc()")
    
    return elapsed_time

def compute_vacf(velocity_trajectory):
    """
    Compute the velocity autocorrelation function (VACF) averaged over all particles.
    Returns: vacf (array, length n_steps+1)
    """
    n_steps = velocity_trajectory.shape[0] - 1

    vacf = np.zeros(n_steps+1)
    v0 = velocity_trajectory[0]  # shape (n_particles, 3)
    for t in range(n_steps+1):
        vt = velocity_trajectory[t]
        # Average dot product over all particles
        vacf[t] = np.mean(np.sum(v0 * vt, axis=1))
    return vacf

#----------------------------------------------------------------
#   P A R A M E T E R S
#----------------------------------------------------------------
# system
n_particles = 400
mass_argon =  39.95             # mass in u = 1e-3 kg/mol
sigma_argon = 0.34              # sigma in nm     Argon: 0.34
epsilon_argon = 120*R*1e-3      # epsilon in kJ/mol Argon: 120

# simulation
dt = 0.1             # ps
n_steps = 1000 
temperature = 300     # K
box_length = 50      # nm
tau_thermostat = 1  # thermostat coupling constant in 1/ps
rij_min = 1e-2      # nm
NVT = False          # switch to decide between NVT and NVE

# output

def run(r_cut=None, cut_smooth=False, neighbor_steps=None, label=nano.generate(size=6), seed=42):

    outDir = f'out/{label}'  
    os.makedirs(outDir, exist_ok=True)
    file_name_base = os.path.join(outDir, "sim")  # file name for all output files

    #----------------------------------------------------------------
    #   P R O G R A M
    #----------------------------------------------------------------
    # start the timer
    tic()

    #
    # initialize simulation parameters
    #
    sim = SimulationParameters(dt = dt, 
                            n_steps = n_steps, 
                            temperature = temperature, 
                            box_length = box_length, 
                            tau_thermostat = tau_thermostat,
                            rij_min=rij_min,
                            r_cut=r_cut,
                            cut_smooth=cut_smooth,
                            neighbor_steps=neighbor_steps,
                            seed=seed
                            )

    #
    # initialize ParticleSystem 
    #
    ps = ParticleSystem(n_particles)
    

    # fill in the parameters for argon
    for i in range(n_particles): 
        ps.set_parameters(i, mass=mass_argon, sigma=sigma_argon, epsilon=epsilon_argon)

    # set initial positions     
    initialize_positions(ps, sim.box_length)

    get_neighbors(ps, sim)

    # set initial velocities     
    initialize_velocities(ps, sim.temperature)

    # calculate force according to initial positions
    calculate_force(ps, sim)

    # calculate box density
    rho = density(ps, sim)

    

    # calculate initial values of variable properties
    E_pot_init = potential_energy(ps, sim)
    E_kin_init = kinetic_energy(ps)
    T_init = instantaneous_temperature(ps)
    P_init = ideal_gas_pressure(ps, sim)


    # initialize position trajectory
    position_trajectory = np.zeros((sim.n_steps+1, n_particles, 3))
    position_trajectory[0,:,:] = ps.position # initial position

    # initialize velocity trajectory
    velocity_trajectory = np.zeros((sim.n_steps+1, n_particles, 3))
    velocity_trajectory[0,:,:] = ps.velocity  # initial velocities

    # initialize energy trajectory
    energy_trajectory = np.zeros((sim.n_steps+1, 4))
    energy_trajectory[0,0] = potential_energy( ps, sim)       # potential energy
    energy_trajectory[0,1] = kinetic_energy(ps)               # kinetic energy
    energy_trajectory[0,2] = instantaneous_temperature(ps)    # instantaneous pressure
    energy_trajectory[0,3] = ideal_gas_pressure(ps, sim)      # ideal gas pressure


    #--------------------------------------------------
    #  The actual MD simulation
    #--------------------------------------------------
    for i in range(sim.n_steps):
        get_neighbors(ps, sim) 
        if NVT==True:
            simulate_NVT_step(ps, sim)
        else: 
            simulate_NVE_step(ps, sim)
            
        # store updated positions
        position_trajectory[i+1,:,:] = ps.position # store updated positions
        
        # store updated velocities
        velocity_trajectory[i+1,:,:] = ps.velocity

        # store updated energies, temperature and pressure
        energy_trajectory[i+1,0] = potential_energy( ps, sim)     # potential energy
        energy_trajectory[i+1,1] = kinetic_energy(ps)             # kinetic energy
        energy_trajectory[i+1,2] = instantaneous_temperature(ps)  # instantaneous pressure
        energy_trajectory[i+1,3] = ideal_gas_pressure(ps, sim)    # ideal gas pressure


    #--------------------------------------
    # W R I T E    T R A J E C T O R I E S 
    #--------------------------------------
    # write position trajectory to file
    write_xyz_trajectory(file_name_base + "_pos.xyz", position_trajectory, atom_symbol="Ar")
    # write energy trajectory to file (binary and text)
    np.save(file_name_base + "_ene.npy", energy_trajectory)
    np.savetxt(file_name_base + "_ene.dat", energy_trajectory, fmt="%.6e", header="#E_pot  E_kin  T  P", comments='')


    #----------------------------------------------------
    # P L O T   E N E R G Y   T R A J E C T O R I E S
    #----------------------------------------------------
    # set time axis
    time_ps = np.arange(sim.n_steps + 1) * sim.dt

    #
    # potential energy
    # 
    E_pot_min = np.mean(energy_trajectory[:,0]) - 1   # lower limit of E_pot axis
    E_pot_max = np.mean(energy_trajectory[:,0]) + 1   # upper limit of E_pot axis 

    plt.figure(figsize=(8, 6))
    plt.plot(time_ps, energy_trajectory[:,0]) 
    plt.ylim(E_pot_min, E_pot_max)
    plt.xlabel("time [ps]", fontsize=14)
    plt.ylabel("E_pot [kJ/mol]", fontsize=14)

    plt.savefig(file_name_base + "_Epot.png", dpi=300, bbox_inches='tight')

    #
    # kinetic energy
    # 
    E_kin_min = np.mean(energy_trajectory[:,1]) - 100   # lower limit of E_kin axis
    E_kin_max = np.mean(energy_trajectory[:,1]) + 100   # upper limit of E_kin axis 

    plt.figure(figsize=(8, 6))
    plt.plot(time_ps, energy_trajectory[:,1]) 
    plt.ylim(E_kin_min, E_kin_max)
    plt.xlabel("time [ps]", fontsize=14)
    plt.ylabel("E_kin [kJ/mol]", fontsize=14)

    plt.savefig(file_name_base + "_Ekin.png", dpi=300, bbox_inches='tight')

    #
    # temperature
    # 
    T_min = np.mean(energy_trajectory[:,2]) - 100   # lower limit of T axis
    T_max = np.mean(energy_trajectory[:,2]) + 100   # upper limit of T axis 

    plt.figure(figsize=(8, 6))
    plt.plot(time_ps, energy_trajectory[:,2]) 
    plt.ylim(T_min, T_max)
    plt.xlabel("time [ps]", fontsize=14)
    plt.ylabel("T [K]", fontsize=14)

    plt.savefig(file_name_base + "_T.png", dpi=300, bbox_inches='tight')

    #
    # pressure
    # 
    P_min = np.mean(energy_trajectory[:,3]) - 200   # lower limit of P axis
    P_max = np.mean(energy_trajectory[:,3]) + 200   # upper limit of P axis 

    plt.figure(figsize=(8, 6))
    plt.plot(time_ps, energy_trajectory[:,3]) 
    plt.ylim(P_min, P_max)
    plt.xlabel("time [ps]", fontsize=14)
    plt.ylabel("P [Pa]", fontsize=14)

    plt.savefig(file_name_base + "_P.png", dpi=300, bbox_inches='tight')

    #
    # Velocity Auto-correlation Function (VACF) 
    #
    vacf = compute_vacf(velocity_trajectory)
    np.save(file_name_base + "_vacf.npy", vacf)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(sim.n_steps+1) * sim.dt, vacf)
    plt.xlabel("Time lag [ps]")
    plt.ylabel("VACF [nm^2/ps^2]")
    plt.title("Velocity Auto-correlation Function")
    plt.savefig(file_name_base + "_VACF.png", dpi=300, bbox_inches='tight')

    # time axis in ps
    time_ps = np.arange(sim.n_steps + 1) * sim.dt  # assuming dt is in ps

    area_vacf = np.trapezoid(vacf, time_ps)  # area under curve

    #--------------------------------------
    # O U T P U T 
    #--------------------------------------
    elapsed_time = toc()   # stop the timer
    output_lines = []

    output_lines.append("")
    output_lines.append("----------------------------------------------------------")
    output_lines.append("Simulation parameters ")    
    output_lines.append("----------------------------------------------------------")
    output_lines.append(f"{'Number of particles:':<30}{ps.n:>10.0f} ")
    output_lines.append(f"{'Box length:':<30}{sim.box_length:>10.3e} nm")
    output_lines.append(f"{'Box volume:':<30}{sim.box_length**3:>10.3e} nm^3")
    output_lines.append(f"{'Density:':<30}{rho:>10.3e} g/cm^3")
    output_lines.append("")   
    output_lines.append(f"{'Time step:':<30}{sim.dt:>10.3f} ps")
    output_lines.append(f"{'Number of time steps:':<30}{sim.n_steps:>10.0f}")
    output_lines.append(f"{'Simulation time:':<30}{sim.n_steps * sim.dt :>10.3e} ps")
    output_lines.append("")   
    if NVT==True: 
        output_lines.append(f"{'Ensemble:':<30}{'NVT':>10}")
        output_lines.append(f"{'Thermostat temperature:':<30}{sim.temperature:>10.0f} K")
        output_lines.append(f"{'Thermostat coupling:':<30}{sim.tau_thermostat:>10.3e} ps")
    else: 
        output_lines.append(f"{'Ensemble:':<30}{'NVE':>10}")
        output_lines.append(f"{'Initial velocities:':<30}{sim.temperature:>10.0f} K")

    output_lines.append("")     
    output_lines.append(f"{'Lower cutoff radius:':<30}{sim.rij_min:>10.3f} nm")
    output_lines.append(f"{'Upper cutoff radius:':<30}{sim.r_cut:>10.3f} nm" if sim.r_cut is not None else f"{'Upper cutoff radius:':<30}{'None':>10}")
    output_lines.append(f"{'Potential Smoothing:':<30}{str(sim.cut_smooth):>10}")
    output_lines.append(f"{'Neighbor steps:':<30}{sim.neighbor_steps:>10}" if sim.neighbor_steps is not None else f"{'Neighbor steps:':<30}{'None':>10}")
    output_lines.append(f"{'VACF area:':<30}{area_vacf:>10.3f} nm^2/ps")

    output_lines.append(f"{'Seed:':<30}{seed:>10}")
    output_lines.append("----------------------------------------------------------")
    if elapsed_time: 
        time_per_time_step = elapsed_time/sim.n_steps
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_lines.append(f"{'Elapsed time:':<30}{elapsed_time:>10.3f} s")   
        output_lines.append(f"{'Elapsed time per time step:':<30}{time_per_time_step:>10.3f} s")
        output_lines.append(f"{'Time stamp:':<30}{now} s")
    output_lines.append("----------------------------------------------------------")
    output_lines.append("END")  
    output_lines.append("----------------------------------------------------------")

    # Print to screen
    for line in output_lines:
        print(line)
    
    # Write to file
    with open(file_name_base + ".out", "w") as f:
        for line in output_lines:
            f.write(line + "\n")


    result = {
        "r_cut": sim.r_cut,
        "cut_smooth": sim.cut_smooth,
        "neighbor_steps": sim.neighbor_steps,
        "seed": seed,
        "avg_E_pot": np.mean(energy_trajectory[:,0]),
        "e_total": energy_trajectory[:, 0] + energy_trajectory[:, 1],  # total energy as float
        "vacf_area": area_vacf,
        "elapsed_time": elapsed_time
    }
    return result    






#----------------------------------------------------------------
#   M A I N
#----------------------------------------------------------------
run_label = nano.generate(size=6) 
r_cuts = np.linspace(1 * sigma_argon, 5 * sigma_argon, 15)
cut_smooths = [False]
neighbor_steps = [1, 5, 10]  # Different neighbor steps to test
seeds = [5934, 4940, 9211, 6274, 8843]  # 5 random seeds for reproducibility
results = []
    
# 1. Run the reference simulation

reference_results = {}
for seed in seeds:
    print(f"Running reference simulation for seed {seed}...")
    ref_res = run(
        r_cut=None,
        cut_smooth=False,
        neighbor_steps=None,
        label=f'run_{run_label}_reference_seed_{seed}',
        seed=seed
    )
    plt.close('all')
    print(f"Reference simulation for seed {seed} completed with results: {ref_res}")
    reference_results[seed] = ref_res

# 2. Parameter sweep
run_counter = 1
total_runs = len(r_cuts) * len(cut_smooths) * len(neighbor_steps) * len(seeds)
for r_cut in r_cuts:
    for cut_smooth in cut_smooths:
        for neighbor_step in neighbor_steps:
            for seed in seeds:
                print(f"Running simulation {run_counter}/{total_runs} with r_cut={r_cut}, cut_smooth={cut_smooth}, neighbor_steps={neighbor_step}, seed={seed}")
                res = run(
                    r_cut=r_cut,
                    cut_smooth=cut_smooth,
                    neighbor_steps=neighbor_step,
                    label='run_' + run_label + f'/r_cut_{r_cut}_cut_smooth_{cut_smooth}_neighbor_steps_{neighbor_step}',
                    seed=seed
                )
                plt.close('all') 
                print(f"Simulation {run_counter}/{total_runs} completed!")
                results.append(res)
                run_counter += 1

# 3. Convert results to DataFrame for easy plotting
df = pd.DataFrame(results)

# Map reference values for correct subtraction
def get_ref_val(seed, key):
    return float(reference_results[seed][key])

df['delta_E_pot'] = np.abs(df.apply(lambda row: row['avg_E_pot'] - get_ref_val(row['seed'], 'avg_E_pot'), axis=1))
# Compute energy drift as the slope of total energy vs. time for each run
def compute_energy_drift(e_total, dt):
    # Linear fit: time vs. e_total
    time = np.arange(len(e_total)) * dt
    slope = np.polyfit(time, e_total, 1)[0]
    return slope

df['E_drift'] = df.apply(lambda row: compute_energy_drift(row['e_total'], dt), axis=1)
df['delta_time'] = df.apply(lambda row: 100 * row['elapsed_time'] / get_ref_val(row['seed'], 'elapsed_time'), axis=1)

# 4a. Aggregate by parameter set (excluding seed), take mean and std
group_cols = ['r_cut', 'cut_smooth', 'neighbor_steps']
agg_cols = ['delta_E_pot', 'delta_time', 'E_drift', 'vacf_area']
agg_df = (
    df[df['r_cut'].notnull()]  # Only parameter sweep, not reference
    .groupby(group_cols)
    [agg_cols]
    .agg(['mean', 'std'])
    .reset_index()
)
# Flatten column names
agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

# 5. Plotting

neighbor_steps_unique = sorted(df['neighbor_steps'].dropna().unique())
color_cycle = plt.cm.tab10.colors
color_map = {ns: color_cycle[i % len(color_cycle)] for i, ns in enumerate(neighbor_steps_unique)}

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
axes = axes.flatten()

ylabels = [
    "Abs. Δ Avg. Potential Energy (kJ/mol)",
    "Time (relative to reference, %)",
    "Energy Drift (kJ/mol/ps)",
    "VACF Area (nm^2/ps)"
]
ycols = [
    "delta_E_pot",
    "delta_time",
    "E_drift",
    "vacf_area"
]

handles_labels = None

# Add a switch to control log scale for x-axis
use_log_scale = True  # Set to False for linear scale

for idx, (ax, ylabel, ycol) in enumerate(zip(axes, ylabels, ycols)):
    for ns_idx, neighbor_step in enumerate(neighbor_steps_unique):
        for cs_idx, cut_smooth in enumerate([False, True]):
            subset = df[
                (df['cut_smooth'] == cut_smooth) &
                (df['neighbor_steps'] == neighbor_step) &
                (df['r_cut'].notnull())
            ].copy()
            subset['r_cut'] = subset['r_cut'].astype(float)
            subset = subset.sort_values('r_cut')
            if not subset.empty:
                # Jitter: scale is a fraction of the r_cut step
                jitter_scale = (r_cuts[1] - r_cuts[0]) * 0.15
                np.random.seed(hash((neighbor_step, cut_smooth, idx)) % (2**32))
                jitter = np.random.uniform(-jitter_scale, jitter_scale, size=len(subset))
                r_cut_jittered = subset['r_cut'].values + jitter

                # r_cut in units of sigma and nm
                r_cut_sigma = r_cut_jittered / sigma_argon
                r_cut_nm = r_cut_jittered

                # Only plot non-negative values
                yvals = subset[ycol].values
                mask = yvals >= 0
                r_cut_sigma = r_cut_sigma[mask]
                r_cut_nm = r_cut_nm[mask]
                yvals = yvals[mask]

                # Slightly less vibrant color
                base_color = np.array(color_map[neighbor_step])
                faded_color = tuple(base_color * 0.6 + 0.4)  # fade toward white

                linestyle = '-' if not cut_smooth else ':'
                label = f"neighbor_steps={neighbor_step}, smooth={'on' if cut_smooth else 'off'}"
                # Plot small points for each seed
                line = ax.plot(
                    r_cut_sigma,
                    yvals,
                    marker='o',
                    linestyle='None',
                    markersize=4,
                    label=label + f" (r_cut={r_cut_nm.min():.2f}-{r_cut_nm.max():.2f} nm)",
                    color=faded_color,
                    alpha=0.7,
                )[0]
                # Plot mean in vibrant color
                mean_vals = []
                mean_r_cut_sigma = []
                for r_cut_val in sorted(subset['r_cut'].unique()):
                    # Mask for current r_cut value and non-negative yvals
                    mask_mean = (subset['r_cut'] == r_cut_val) & (subset[ycol] >= 0)
                    y_mean = subset.loc[mask_mean, ycol].mean() if mask_mean.any() else np.nan
                    if not np.isnan(y_mean):
                        mean_vals.append(y_mean)
                        mean_r_cut_sigma.append(r_cut_val / sigma_argon)
                if mean_vals:
                    ax.plot(
                        mean_r_cut_sigma,
                        mean_vals,
                        marker='s',
                        linestyle=linestyle,
                        markersize=7,
                        color=color_map[neighbor_step],
                        label=label + " (mean)",
                        alpha=1.0,
                        zorder=10
                    )
                # Save handles and labels from the first subplot only
                if idx == 0 and handles_labels is None:
                    handles_labels = ([], [])
                if idx == 0:
                    handles_labels[0].append(line)
                    handles_labels[1].append(label + f" (r_cut={r_cut_nm.min():.2f}-{r_cut_nm.max():.2f} nm)")
    ax.set_xlabel("Cutoff radius (σ units)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    # Optionally, show secondary x-axis for nm
    def sigma_to_nm(x):
        return x * sigma_argon
    def nm_to_sigma(x):
        return x / sigma_argon
    secax = ax.secondary_xaxis('top', functions=(sigma_to_nm, nm_to_sigma))
    secax.set_xlabel("Cutoff radius (nm)")
    # Set y-axis scale: log for all except delta_time
    if use_log_scale and ycol != 'delta_time':
        ax.set_yscale('log', nonpositive='mask')
        secax.set_yscale('log', nonpositive='mask')
    else:
        ax.set_yscale('linear')
        secax.set_yscale('linear')

# Hide the last empty subplot if not needed
if len(ycols) < len(axes):
    for ax in axes[len(ycols):]:
        ax.axis('off')

# Add legend to separate figure
if handles_labels is not None:
    # Create a new figure for the legend only
    legend_fig = plt.figure(figsize=(4, 2))
    legend_fig.legend(
        handles_labels[0], handles_labels[1],
        loc='center', fontsize=12
    )
    legend_fig.tight_layout()
    legend_fig.savefig('out/run_' + run_label + '/summary_legend.png', dpi=400, bbox_inches='tight')
    plt.close(legend_fig)

plt.suptitle("Effect of Cutoff, Neighbor Steps, and Smoothing (relative to reference)", fontsize=16)
plt.tight_layout()
plt.savefig('out/run_' + run_label + '/summary_all_vs_r_cut_relative.png', dpi=400, bbox_inches='tight')
plt.show()
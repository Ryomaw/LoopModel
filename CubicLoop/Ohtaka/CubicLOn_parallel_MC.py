"""Parallel Monte Carlo simulation for the cubic loop O(n) model on Ohtaka.

This module extends CubicLOn_rev_MC.py with parallel execution capabilities
for large-scale parameter sweeps using multiprocessing.
"""

import csv
import math
import time
import sys
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HORIZONTAL = 0
VERTICAL = 1
WINDING_SECTOR_LABELS = ("even_even", "even_odd", "odd_even", "odd_odd")


# ============================================================================
# Import core functions from CubicLOn_rev_MC (or duplicate minimal versions)
# ============================================================================

def _format_elapsed(seconds):
    """Return a compact elapsed-time label."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def initialize_bond_configuration(lattice_size, seed=None):
    """Return the empty bond configuration and RNG."""
    rng = np.random.default_rng(seed)
    bond = np.zeros((lattice_size, lattice_size, 2), dtype=np.int8)
    return bond, rng


def flip_line(bond, line_index, direction):
    """Flip all bond numbers on one of the 2L straight line sets."""
    if direction == HORIZONTAL:
        bond[:, line_index, HORIZONTAL] ^= 1
    elif direction == VERTICAL:
        bond[line_index, :, VERTICAL] ^= 1
    else:
        raise ValueError("direction must be 0 (horizontal) or 1 (vertical).")


def flip_plaquette(bond, x, y):
    """Flip the four bonds surrounding the plaquette with lower-left corner (x, y)."""
    lattice_size = bond.shape[0]
    bond[x, y, HORIZONTAL] ^= 1
    bond[x, y, VERTICAL] ^= 1
    bond[x, (y + 1) % lattice_size, HORIZONTAL] ^= 1
    bond[(x + 1) % lattice_size, y, VERTICAL] ^= 1


def apply_move(bond, move):
    """Apply a proposed update in place."""
    move_type = move[0]
    if move_type == "line":
        _, direction, line_index = move
        flip_line(bond, line_index, direction)
    elif move_type == "plaquette":
        _, x, y = move
        flip_plaquette(bond, x, y)
    else:
        raise ValueError(f"Unknown move type: {move_type}")


def propose_move(lattice_size, rng, line_update_rate=0.1):
    """Generate one of the user-specified update proposals."""
    if rng.random() < line_update_rate:
        line_id = int(rng.integers(2 * lattice_size))
        if line_id < lattice_size:
            return ("line", HORIZONTAL, line_id)
        return ("line", VERTICAL, line_id - lattice_size)
    x = int(rng.integers(lattice_size))
    y = int(rng.integers(lattice_size))
    return ("plaquette", x, y)


def occupied_bond_count(bond):
    """Return the total number of occupied bonds."""
    return int(np.sum(bond))


def plaquette_operator(bond, x, y):
    """Return P_{x,y} for the plaquette whose lower-left corner is (x, y)."""
    lattice_size = bond.shape[0]
    x_index = int(x) % lattice_size
    y_index = int(y) % lattice_size
    if (
        bond[x_index, y_index, HORIZONTAL]
        and bond[x_index, y_index, VERTICAL]
        and bond[x_index, (y_index + 1) % lattice_size, HORIZONTAL]
        and bond[(x_index + 1) % lattice_size, y_index, VERTICAL]
    ):
        return 1
    return 0


def plaquette_order_parameters(bond):
    """Return the order parameters V_x and V_y from the plaquette field P."""
    lattice_size = bond.shape[0]
    site_count = float(lattice_size ** 2)
    vx_sum = 0.0
    vy_sum = 0.0
    for x in range(lattice_size):
        sign_x = -1.0 if (x % 2) else 1.0
        for y in range(lattice_size):
            p_xy = plaquette_operator(bond, x, y)
            if p_xy:
                sign_y = -1.0 if (y % 2) else 1.0
                vx_sum += sign_x
                vy_sum += sign_y
    return 4.0 * vx_sum / site_count, 4.0 * vy_sum / site_count


def horizontal_cut_bond_count(bond, y):
    """Return the number of occupied bonds crossing y + 1/2."""
    lattice_size = bond.shape[0]
    y_index = int(y) % lattice_size
    return int(np.sum(bond[:, y_index, VERTICAL]))


def horizontal_cut_bond_parity(bond, y):
    """Return the parity of the occupied-bond count on the cut y + 1/2."""
    crossing_count = horizontal_cut_bond_count(bond, y)
    if crossing_count % 2 == 0:
        return "even"
    return "odd"


def vertical_cut_bond_count(bond, x):
    """Return the number of occupied bonds crossing x + 1/2."""
    lattice_size = bond.shape[0]
    x_index = int(x) % lattice_size
    return int(np.sum(bond[x_index, :, HORIZONTAL]))


def vertical_cut_bond_parity(bond, x):
    """Return the parity of the occupied-bond count on the cut x + 1/2."""
    crossing_count = vertical_cut_bond_count(bond, x)
    if crossing_count % 2 == 0:
        return "even"
    return "odd"


def winding_sector_key(bond, x_cut=0, y_cut=0):
    """Return the parity sector key for the pair (w_x, w_y)."""
    wx_parity = horizontal_cut_bond_parity(bond, y_cut)
    wy_parity = vertical_cut_bond_parity(bond, x_cut)
    return f"{wx_parity}_{wy_parity}"


def initialize_winding_sector_counts():
    """Return a zero-initialized counter dictionary for winding sectors."""
    return {key: 0 for key in WINDING_SECTOR_LABELS}


def vertex_degree(bond, x, y):
    """Return the degree of vertex (x, y) in the occupied-bond graph."""
    lattice_size = bond.shape[0]
    degree = 0
    degree += int(bond[x, y, HORIZONTAL])
    degree += int(bond[x, y, VERTICAL])
    degree += int(bond[(x - 1) % lattice_size, y, HORIZONTAL])
    degree += int(bond[x, (y - 1) % lattice_size, VERTICAL])
    return degree


def is_even_subgraph(bond):
    """Check whether every vertex has even degree."""
    lattice_size = bond.shape[0]
    for x in range(lattice_size):
        for y in range(lattice_size):
            if vertex_degree(bond, x, y) % 2 != 0:
                return False
    return True


def assert_even_subgraph(bond):
    """Raise an error if the configuration violates the even-subgraph constraint."""
    lattice_size = bond.shape[0]
    for x in range(lattice_size):
        for y in range(lattice_size):
            degree = vertex_degree(bond, x, y)
            if degree % 2 != 0:
                raise ValueError(
                    f"Even-subgraph condition violated at vertex ({x}, {y}) with degree {degree}."
                )


def build_adjacency(bond):
    """Build adjacency information for occupied bonds."""
    lattice_size = bond.shape[0]
    adjacency = {}

    def add_edge(site_a, site_b):
        adjacency.setdefault(site_a, []).append(site_b)
        adjacency.setdefault(site_b, []).append(site_a)

    for x in range(lattice_size):
        for y in range(lattice_size):
            if bond[x, y, HORIZONTAL]:
                add_edge((x, y), ((x + 1) % lattice_size, y))
            if bond[x, y, VERTICAL]:
                add_edge((x, y), (x, (y + 1) % lattice_size))

    return adjacency


def connected_component_count(bond):
    """Return the total number of clusters c, including isolated vertices."""
    adjacency = build_adjacency(bond)
    visited = set()
    component_count = 0

    for site in adjacency:
        if site in visited:
            continue
        component_count += 1
        stack = [site]
        visited.add(site)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

    lattice_size = bond.shape[0]
    isolated_vertex_count = lattice_size ** 2 - len(adjacency)
    return component_count + isolated_vertex_count


def configuration_weight_ratio(old_length, old_components, new_length, new_components, x_weight, n_weight):
    """Return W_new / W_old for the model weight x^L n^c."""
    delta_length = new_length - old_length
    delta_components = new_components - old_components

    if x_weight < 0.0 or n_weight <= 0.0:
        raise ValueError("x_weight must be non-negative and n_weight must be positive.")

    if x_weight == 0.0:
        if new_length > 0 and old_length == 0:
            return 0.0
        if new_length == 0 and old_length > 0:
            return math.inf
        if new_length == 0 and old_length == 0:
            return math.exp(delta_components * math.log(n_weight))
        return 0.0

    return math.exp(
        delta_length * math.log(x_weight) + delta_components * math.log(n_weight)
    )


def metropolis_update(bond, rng, x_weight, n_weight=1.0, line_update_rate=0.1):
    """Perform one Metropolis update using the specified proposal rule."""
    assert_even_subgraph(bond)

    proposed_bond = np.array(bond, copy=True)
    move = propose_move(proposed_bond.shape[0], rng, line_update_rate=line_update_rate)
    apply_move(proposed_bond, move)
    assert_even_subgraph(proposed_bond)

    old_length = occupied_bond_count(bond)
    new_length = occupied_bond_count(proposed_bond)
    old_components = connected_component_count(bond)
    new_components = connected_component_count(proposed_bond)

    ratio = configuration_weight_ratio(
        old_length,
        old_components,
        new_length,
        new_components,
        x_weight,
        n_weight,
    )
    accepted = ratio >= 1.0 or rng.random() < ratio

    if accepted:
        bond[:, :, :] = proposed_bond

    assert_even_subgraph(bond)
    return bond, accepted, move


def jackknife_mean_and_error(samples, block_size):
    """Estimate the mean and its jackknife error from blocked data."""
    sample_count = len(samples)

    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if sample_count < block_size:
        raise ValueError("block_size must not exceed the number of samples.")

    block_count = sample_count // block_size
    if block_count < 2:
        raise ValueError("At least two jackknife blocks are required.")

    trimmed_count = block_count * block_size
    trimmed_samples = np.asarray(samples[:trimmed_count], dtype=np.float64)
    
    block_means = trimmed_samples.reshape(block_count, block_size).mean(axis=1)
    global_mean = float(np.mean(trimmed_samples))

    leave_one_out_means = (block_count * global_mean - block_means) / (block_count - 1)
    jk_mean = float(np.mean(leave_one_out_means))
    jk_error = math.sqrt(
        (block_count - 1) * np.mean((leave_one_out_means - jk_mean) ** 2)
    )
    return global_mean, float(jk_error), block_count, trimmed_count


def jackknife_heat_capacity(lengths, block_size, lattice_size):
    """Estimate h = (<l^2> - <l>^2) / L^2 and its jackknife error."""
    sample_count = len(lengths)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if sample_count < block_size:
        raise ValueError("block_size must not exceed the number of samples.")

    block_count = sample_count // block_size
    if block_count < 2:
        raise ValueError("At least two jackknife blocks are required.")

    trimmed_count = block_count * block_size
    trimmed_lengths = np.asarray(lengths[:trimmed_count], dtype=np.float64)
    reshaped = trimmed_lengths.reshape(block_count, block_size)

    global_mean = float(np.mean(trimmed_lengths))
    global_second_moment = float(np.mean(trimmed_lengths ** 2))
    heat_capacity = (global_second_moment - global_mean ** 2) / float(lattice_size ** 2)

    block_means = reshaped.mean(axis=1)
    block_second_moments = (reshaped ** 2).mean(axis=1)
    leave_one_out_means = (block_count * global_mean - block_means) / (block_count - 1)
    leave_one_out_second_moments = (
        block_count * global_second_moment - block_second_moments
    ) / (block_count - 1)
    leave_one_out_h = (
        leave_one_out_second_moments - leave_one_out_means ** 2
    ) / float(lattice_size ** 2)

    jk_mean = float(np.mean(leave_one_out_h))
    jk_error = math.sqrt((block_count - 1) * np.mean((leave_one_out_h - jk_mean) ** 2))
    return float(heat_capacity), float(jk_error)


def jackknife_variance_density(samples, block_size, lattice_size):
    """Estimate (<x^2> - <x>^2) / L^2 and its jackknife error."""
    sample_count = len(samples)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if sample_count < block_size:
        raise ValueError("block_size must not exceed the number of samples.")

    block_count = sample_count // block_size
    if block_count < 2:
        raise ValueError("At least two jackknife blocks are required.")

    trimmed_count = block_count * block_size
    trimmed_samples = np.asarray(samples[:trimmed_count], dtype=np.float64)
    reshaped = trimmed_samples.reshape(block_count, block_size)

    global_mean = float(np.mean(trimmed_samples))
    global_second_moment = float(np.mean(trimmed_samples ** 2))
    variance_density = (global_second_moment - global_mean ** 2) / float(lattice_size ** 2)

    block_means = reshaped.mean(axis=1)
    block_second_moments = (reshaped ** 2).mean(axis=1)
    leave_one_out_means = (block_count * global_mean - block_means) / (block_count - 1)
    leave_one_out_second_moments = (
        block_count * global_second_moment - block_second_moments
    ) / (block_count - 1)
    leave_one_out_values = (
        leave_one_out_second_moments - leave_one_out_means ** 2
    ) / float(lattice_size ** 2)

    jk_mean = float(np.mean(leave_one_out_values))
    jk_error = math.sqrt((block_count - 1) * np.mean((leave_one_out_values - jk_mean) ** 2))
    return float(variance_density), float(jk_error)


def measure_observables(
    lattice_size,
    x_weight,
    n_weight=1.0,
    thermalization=1000,
    observation=5000,
    block_size=100,
    line_update_rate=0.1,
    seed=None,
):
    """Estimate moments of l and c for fixed x and n by Monte Carlo sampling."""
    bond, rng = initialize_bond_configuration(lattice_size, seed=seed)

    # Thermalization phase
    for step in range(thermalization):
        metropolis_update(
            bond,
            rng,
            x_weight=x_weight,
            n_weight=n_weight,
            line_update_rate=line_update_rate,
        )

    # Observation phase
    lengths = np.zeros(observation, dtype=np.float64)
    cluster_counts = np.zeros(observation, dtype=np.float64)
    length_per_cluster = np.zeros(observation, dtype=np.float64)
    plaquette_order_x = np.zeros(observation, dtype=np.float64)
    plaquette_order_y = np.zeros(observation, dtype=np.float64)
    accepted_moves = 0
    winding_sector_counts = initialize_winding_sector_counts()

    for step in range(observation):
        _, accepted, _ = metropolis_update(
            bond,
            rng,
            x_weight=x_weight,
            n_weight=n_weight,
            line_update_rate=line_update_rate,
        )
        if accepted:
            accepted_moves += 1
        lengths[step] = occupied_bond_count(bond)
        cluster_counts[step] = connected_component_count(bond)
        if cluster_counts[step] > 0:
            length_per_cluster[step] = lengths[step] / cluster_counts[step]
        else:
            length_per_cluster[step] = 0.0
        vx_value, vy_value = plaquette_order_parameters(bond)
        plaquette_order_x[step] = vx_value
        plaquette_order_y[step] = vy_value
        winding_sector_counts[winding_sector_key(bond)] += 1

    length_squared = lengths ** 2
    length_cubed = lengths ** 3
    length_fourth = lengths ** 4
    cluster_squared = cluster_counts ** 2
    cluster_cubed = cluster_counts ** 3
    cluster_fourth = cluster_counts ** 4

    average_length, length_jackknife_error, block_count, effective_samples = jackknife_mean_and_error(lengths, block_size)
    average_length_squared, length_squared_jackknife_error, _, _ = jackknife_mean_and_error(length_squared, block_size)
    average_length_cubed, length_cubed_jackknife_error, _, _ = jackknife_mean_and_error(length_cubed, block_size)
    average_length_fourth, length_fourth_jackknife_error, _, _ = jackknife_mean_and_error(length_fourth, block_size)
    average_length_per_cluster, length_per_cluster_jackknife_error, _, _ = jackknife_mean_and_error(length_per_cluster, block_size)
    average_plaquette_order_x, plaquette_order_x_jackknife_error, _, _ = jackknife_mean_and_error(
        plaquette_order_x, block_size
    )
    average_plaquette_order_y, plaquette_order_y_jackknife_error, _, _ = jackknife_mean_and_error(
        plaquette_order_y, block_size
    )
    heat_capacity_like, heat_capacity_like_jackknife_error = jackknife_heat_capacity(
        lengths, block_size, lattice_size
    )
    average_cluster_count, cluster_jackknife_error, _, _ = jackknife_mean_and_error(cluster_counts, block_size)
    average_cluster_squared, cluster_squared_jackknife_error, _, _ = jackknife_mean_and_error(cluster_squared, block_size)
    average_cluster_cubed, cluster_cubed_jackknife_error, _, _ = jackknife_mean_and_error(cluster_cubed, block_size)
    average_cluster_fourth, cluster_fourth_jackknife_error, _, _ = jackknife_mean_and_error(cluster_fourth, block_size)
    cluster_heat_capacity_like, cluster_heat_capacity_like_jackknife_error = jackknife_variance_density(
        cluster_counts, block_size, lattice_size
    )

    result = {
        "x": x_weight,
        "n": n_weight,
        "lattice_size": lattice_size,
        "thermalization": thermalization,
        "observation": observation,
        "average_length": average_length,
        "length_jackknife_error": length_jackknife_error,
        "average_length_squared": average_length_squared,
        "length_squared_jackknife_error": length_squared_jackknife_error,
        "average_length_cubed": average_length_cubed,
        "length_cubed_jackknife_error": length_cubed_jackknife_error,
        "average_length_fourth": average_length_fourth,
        "length_fourth_jackknife_error": length_fourth_jackknife_error,
        "average_length_per_cluster": average_length_per_cluster,
        "length_per_cluster_jackknife_error": length_per_cluster_jackknife_error,
        "average_plaquette_order_x": average_plaquette_order_x,
        "plaquette_order_x_jackknife_error": plaquette_order_x_jackknife_error,
        "average_plaquette_order_y": average_plaquette_order_y,
        "plaquette_order_y_jackknife_error": plaquette_order_y_jackknife_error,
        "heat_capacity_like": heat_capacity_like,
        "heat_capacity_like_jackknife_error": heat_capacity_like_jackknife_error,
        "average_cluster_count": average_cluster_count,
        "cluster_jackknife_error": cluster_jackknife_error,
        "average_cluster_squared": average_cluster_squared,
        "cluster_squared_jackknife_error": cluster_squared_jackknife_error,
        "average_cluster_cubed": average_cluster_cubed,
        "cluster_cubed_jackknife_error": cluster_cubed_jackknife_error,
        "average_cluster_fourth": average_cluster_fourth,
        "cluster_fourth_jackknife_error": cluster_fourth_jackknife_error,
        "cluster_heat_capacity_like": cluster_heat_capacity_like,
        "cluster_heat_capacity_like_jackknife_error": cluster_heat_capacity_like_jackknife_error,
        "acceptance_rate": accepted_moves / float(observation),
        "jackknife_block_size": block_size,
        "jackknife_block_count": block_count,
        "effective_samples": effective_samples,
    }
    for key in WINDING_SECTOR_LABELS:
        result[f"winding_sector_{key}_count"] = winding_sector_counts[key]
        result[f"winding_sector_{key}_fraction"] = (
            winding_sector_counts[key] / float(observation)
        )
    return result


# ============================================================================
# Parallel execution wrapper and result aggregation
# ============================================================================

def run_single_simulation(args: Tuple) -> dict:
    """Run a single simulation with the given parameters."""
    lattice_size, n_weight, x_weight, thermalization, observation, seed = args
    return measure_observables(
        lattice_size=lattice_size,
        x_weight=x_weight,
        n_weight=n_weight,
        thermalization=thermalization,
        observation=observation,
        block_size=100,
        line_update_rate=0.1,
        seed=seed,
    )


def run_parallel_sweep(
    lattice_size: int,
    n_values: List[float],
    x_values: List[float],
    ts_os_configs: List[Tuple[int, int]],
    output_dir: str,
    num_workers: int = None,
) -> List[dict]:
    """
    Run parallel Monte Carlo sweep across all parameter combinations.

    Args:
        lattice_size: Lattice size L
        n_values: List of n values
        x_values: List of x coupling values
        ts_os_configs: List of (thermalization, observation) tuples
        output_dir: Directory for results
        num_workers: Number of parallel workers (default: cpu_count())

    Returns:
        List of result dictionaries
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate all parameter combinations
    tasks = []
    seed_counter = 1000
    for n in n_values:
        for x in x_values:
            for ts, os in ts_os_configs:
                tasks.append((lattice_size, n, x, ts, os, seed_counter))
                seed_counter += 1

    print(f"[INFO] Total tasks: {len(tasks)}", flush=True)
    print(f"[INFO] Using {num_workers} workers", flush=True)

    start_time = time.time()
    results = []

    with Pool(processes=num_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(run_single_simulation, tasks)):
            results.append(result)
            elapsed = _format_elapsed(time.time() - start_time)
            print(
                f"[{idx + 1}/{len(tasks)}] Completed L={result['lattice_size']}, "
                f"n={result['n']:g}, x={result['x']:.2f}, "
                f"ts={result['thermalization']}, os={result['observation']} "
                f"({elapsed})",
                flush=True,
            )

    total_time = _format_elapsed(time.time() - start_time)
    print(f"[INFO] All tasks completed in {total_time}", flush=True)

    return results


def save_results_to_csv(results: List[dict], output_path: str):
    """Save measured observables to a CSV file."""
    fieldnames = [
        "lattice_size", "n", "x", "thermalization", "observation",
        "average_length", "length_jackknife_error",
        "average_length_squared", "length_squared_jackknife_error",
        "average_length_cubed", "length_cubed_jackknife_error",
        "average_length_fourth", "length_fourth_jackknife_error",
        "average_length_per_cluster", "length_per_cluster_jackknife_error",
        "average_plaquette_order_x", "plaquette_order_x_jackknife_error",
        "average_plaquette_order_y", "plaquette_order_y_jackknife_error",
        "heat_capacity_like", "heat_capacity_like_jackknife_error",
        "average_cluster_count", "cluster_jackknife_error",
        "average_cluster_squared", "cluster_squared_jackknife_error",
        "average_cluster_cubed", "cluster_cubed_jackknife_error",
        "average_cluster_fourth", "cluster_fourth_jackknife_error",
        "cluster_heat_capacity_like", "cluster_heat_capacity_like_jackknife_error",
        "acceptance_rate", "jackknife_block_size", "jackknife_block_count", "effective_samples",
        "winding_sector_even_even_count", "winding_sector_even_odd_count",
        "winding_sector_odd_even_count", "winding_sector_odd_odd_count",
        "winding_sector_even_even_fraction", "winding_sector_even_odd_fraction",
        "winding_sector_odd_even_fraction", "winding_sector_odd_odd_fraction",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result[name] for name in fieldnames})


def main():
    """Main entry point for parallel sweep."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel Monte Carlo simulation for cubic loop O(n) model"
    )
    parser.add_argument(
        "--lattice_size", type=int, default=16,
        help="Lattice size L (default: 16)"
    )
    parser.add_argument(
        "--n_values", type=float, nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100],
        help="List of n values (default: 1 2 3 ... 100)"
    )
    parser.add_argument(
        "--x_min", type=float, default=0.1,
        help="Minimum x value (default: 0.1)"
    )
    parser.add_argument(
        "--x_max", type=float, default=5.0,
        help="Maximum x value (default: 5.0)"
    )
    parser.add_argument(
        "--x_step", type=float, default=0.1,
        help="x step size (default: 0.1)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results_L16",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count - 1)"
    )

    args = parser.parse_args()

    # Generate x values
    x_values = list(np.arange(args.x_min, args.x_max + args.x_step / 2, args.x_step))
    x_values = [round(x, 2) for x in x_values]  # Avoid floating point errors

    # ts and os configurations
    ts_os_configs = [
        (10000, 40000),
        (40000, 40000),
        (10000, 160000),
        (40000, 160000),
    ]

    print("=" * 80, flush=True)
    print("Parallel Monte Carlo Simulation for Cubic Loop O(n) Model", flush=True)
    print("=" * 80, flush=True)
    print(f"Lattice size L = {args.lattice_size}", flush=True)
    print(f"n values: {args.n_values}", flush=True)
    print(f"x values: {args.x_min} to {args.x_max} (step {args.x_step})", flush=True)
    print(f"Number of x points: {len(x_values)}", flush=True)
    print(f"ts/os configurations: {ts_os_configs}", flush=True)
    print(f"Total simulations: {len(args.n_values) * len(x_values) * len(ts_os_configs)}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print("=" * 80, flush=True)

    # Run parallel sweep
    results = run_parallel_sweep(
        lattice_size=args.lattice_size,
        n_values=args.n_values,
        x_values=x_values,
        ts_os_configs=ts_os_configs,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )

    # Save results
    csv_path = Path(args.output_dir) / "results.csv"
    save_results_to_csv(results, str(csv_path))
    print(f"\n[INFO] Results saved to {csv_path}", flush=True)


if __name__ == "__main__":
    main()

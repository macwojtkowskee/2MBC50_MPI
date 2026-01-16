from mpi4py import MPI
import numpy as np  
import time
import matplotlib.pyplot as plt

def apply_periodic_boundary_conditions(positions, box_size):
    """
    Apply periodic boundary conditions to particle positions.
    Wraps positions into the box of size `box_size` using modulo operation.
    """
    return positions % box_size

def minimum_image_convention(delta_r, box_size):
    """
    Apply the minimum image convention for periodic boundary conditions.
    Adjust the displacement vector delta_r to ensure it represents the shortest distance.
    """
    return delta_r - box_size * np.round(delta_r / box_size)

def LJ_force_magnitude(r,epsilon,sigma,box_size):
    r6 = (sigma / r) ** 6
    r12 = r6 * r6
    return 48 * epsilon * (r12 - 0.5 * r6) / r*r
    
def compute_accelerations(positions, masses, epsilon, sigma, box_size, cutoff, local_indices=None):
    """
    Compute accelerations acting on particles using the Lennard-Jones potential
    and apply periodic boundary conditions.
    """
    N = len(positions)
    accelerations = np.zeros_like(positions)
    
    if local_indices is None:
        local_indices = np.arange(N)
    
    for i in local_indices:
        for j in range(i + 1, N):
            delta_r = positions[j] - positions[i]
            delta_r = minimum_image_convention(delta_r, box_size)
            r2 = np.dot(delta_r, delta_r)
            r = np.sqrt(r2)

            if r > 0 and r < cutoff:
                force = delta_r * LJ_force_magnitude(r,epsilon,sigma,box_size)

                accelerations[i] += force / masses[i]
                accelerations[j] -= force / masses[j]

    return accelerations

def velocity_verlet(positions, velocities, masses, dt, steps, box_size, epsilon=1.0, sigma=1.0, cutoff=1.0):
    """
    MPI parallelized Velocity Verlet algorithm with periodic boundary conditions.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
  
    N = len(positions)
    
    sizes = [N // size + (1 if i < N % size else 0) for i in range(size)]
    offsets = [sum(sizes[:i]) for i in range(size)]
    start = offsets[rank]
    local_size = sizes[rank]

    recv_counts = np.array(sizes) * 3
    recv_displacements = np.array(offsets) * 3

    local_indices = np.arange(start, start + local_size)

    f = open("argon.xyz", "w")

    accelerations = np.zeros_like(positions)

    for step in range(steps):

        if rank == 0:
            f.write(str(N)+"\n"+"Argon"+"\n")
            for i in range(N):
                f.write("Ar "+ str(10.0*positions[i,0]) + " " + str(10.0*positions[i,1]) + " " + str(10.0*positions[i,2]) +"\n")

        comm.Bcast(positions, root=0)

        local_accelerations = np.zeros_like(positions)
        local_part = compute_accelerations(positions, masses, epsilon, sigma, box_size, cutoff, local_indices)
        
        comm.Allreduce(local_part, local_accelerations, op=MPI.SUM)
        accelerations = local_accelerations

        positions += velocities * dt + 0.5 * accelerations * dt * dt
        positions = apply_periodic_boundary_conditions(positions, box_size)
        
        comm.Bcast(positions, root=0)
        
        local_accelerations_new = np.zeros_like(positions)
        local_part_new = compute_accelerations(positions, masses, epsilon, sigma, box_size, cutoff, local_indices)
        
        comm.Allreduce(local_part_new, local_accelerations_new, op=MPI.SUM)
        accelerations_new = local_accelerations_new
        
        velocities += 0.5 * (accelerations + accelerations_new) * dt
        accelerations = accelerations_new
    
    if rank == 0:
        f.close()
        return positions, velocities
    return None, None

import random
import math

def generate_random_points_in_box_3d(box_size, num_points, min_distance):
    
    points = []

    def is_valid_point(new_point):
        for point in points:
            if math.dist(new_point, point) < min_distance:
                return False
        return True

    while len(points) < num_points:
        x = random.uniform(0, box_size)
        y = random.uniform(0, box_size)
        z = random.uniform(0, box_size)
        new_point = (x, y, z)
        if is_valid_point(new_point):
            points.append(new_point)

    return points

def plot_execution_times_mpi():
    """
    MPI version: Plot execution times vs number of particles.
    Only rank 0 performs timing and plotting.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # LJ parameters for Argon
    eps_argon = 0.997
    argon_sigma = 0.34
    dt = 0.0001  # t-step
    steps = 100  # of steps (reduced for faster testing)

    base_N = 100
    base_box = 5.0
    desired_density = base_N / (base_box ** 3)

    # Different box sizes for testing
    box_sizes = [5, 6, 7, 8, 9, 10]
    N_values = []
    execution_times = []

    for box_size in box_sizes:
        N = int(round(desired_density * (box_size ** 3)))
        N_values.append(N)
        cutoff = box_size * 5 / 12
        
        if rank == 0:
            print(f"Running MPI simulation: box_size={box_size}, N={N}, processes={size}")
        if rank == 0:
            positions = np.array(generate_random_points_in_box_3d(box_size, N, 0.4))
            velocities = np.zeros((N, 3))
            masses = 40 * np.ones(N)
        else:
            positions = None
            velocities = None
            masses = None
        
        positions = comm.bcast(positions, root=0)
        velocities = comm.bcast(velocities, root=0)
        masses = comm.bcast(masses, root=0)

        # Synchronize before timing
        comm.Barrier()
        start_time = time.time()
        
        final_positions, final_velocities = velocity_verlet(
            positions, velocities, masses, dt, steps, box_size,
            epsilon=eps_argon, sigma=argon_sigma, cutoff=cutoff,
            rank=rank, size=size)
        
        comm.Barrier()
        elapsed_time = time.time() - start_time
        
        if rank == 0:
            execution_times.append(elapsed_time)
            print(f"--- {elapsed_time} seconds ---\n")
    
    # Only rank 0 creates the plot
    if rank == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(N_values, execution_times, 'bo-', markersize=8, linewidth=2)
        plt.xlabel('Number of particles', fontsize=12)
        plt.ylabel('Execution time (s)', fontsize=12)
        plt.title(f'MPI Execution Time vs Number of Particles (processes={size})', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        
        filename = f'execution_time_vs_N_mpi_{size}procs.png'
        plt.savefig(filename)
        print(f"\nPlot saved as {filename}")
        plt.show()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    N = 800
    box_size = 10.0
    eps_argon = 0.997
    sigma_argon = 0.34
    dt = 0.1
    steps = 1000
    cutoff = 2.4

    if rank == 0:
        positions = np.array(generate_random_points_in_box_3d(box_size, N, 0.4))
        velocities = np.zeros((N, 3))
        masses = 40*np.ones(N)
        start_time = time.time()
    else:
        positions = None
        velocities = None
        masses = None
        start_time = 0
    
    positions = comm.bcast(positions, root=0)
    velocities = comm.bcast(velocities, root=0)
    masses = comm.bcast(masses, root=0)

    final_positions, final_velocities = velocity_verlet(
        positions, velocities, masses, dt, steps, box_size, epsilon=eps_argon, sigma=sigma_argon, cutoff=cutoff)

    if rank == 0:
        print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
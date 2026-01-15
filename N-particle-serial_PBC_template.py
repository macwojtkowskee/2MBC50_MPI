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
    
def compute_accelerations(positions, masses, epsilon, sigma, box_size, cutoff):
    """
    Compute accelerations acting on particles using the Lennard-Jones potential
    and apply periodic boundary conditions.
    """
    N = len(positions)
    accelerations = np.zeros_like(positions)
    
    # ===== ASSIGNMENT 1.1  ===========================================================
    # Implement calculation of the forces on the paricles, then convert to
    # accelerations. Take into account:
    # 1) LJ-interactions are pairwise interactions. Construct a loop
    #    over all pairs (i,j). Can you use some symmetries for this?
    # 2) The strength of the LJ-interactions depend only on the pair-distance. 
    #    What is the correct distance when using Periodic Boundary Conditions?
    # 3) LJ-interactions decrease 'quickly' with pair-distance. Due to the 
    #    PBCs, you also have to avoid double counting of interactions. Both are
    #    usually accounted for by introducing a cutoff distance = maximal distance
    #    for which interactions are calculated. What is the maximal cutoff distance
    #    which makes sense from the geometry of the simulation box? What might be a 
    #    way to find an 'optimal/practical' cutoff value?
    #
    #   INPUT: - positions as an Nx3 array
    #          - masses as an N array
    #          - epsilon, sigma, boxsize, cutoff as real
    
    for i in range(N): 
        for j in range(i+1, N):
            
            distance_vector = minimum_image_convention(positions[i]-positions[j], box_size)            
            distance_value = np.linalg.norm(distance_vector)

            if distance_value < cutoff: 
                force = LJ_force_magnitude(distance_value, epsilon, sigma, box_size) * distance_vector

                accelerations[i] += force/masses[i]
                accelerations[j] -= force/masses[j]
    
    return accelerations

def velocity_verlet(positions, velocities, masses, dt, steps, box_size, epsilon=1.0, sigma=1.0, cutoff=1.0):
    """
    Serial implementation of the Velocity Verlet algorithm with periodic boundary conditions.
    """
    N = len(positions)
    accelerations = compute_accelerations(positions, masses, epsilon, sigma, box_size, cutoff)

    # ========== ASSIGNMENT 1.2 ===========================
    # Make a sanity check for the cutoff for the LJ interactions.
    # use 
    # "assert cutoff [SOMETHING]"  
    
    # Sanity check to verify
    assert cutoff < box_size / 2, "Cutoff must be at most half the box size to avoid double counting"

    #write xyz-file
    f = open("argon.xyz", "w")

    for step in range(steps):

        f.write(str(N)+"\n"+"Argon"+"\n")
        for i in range(N):
            f.write("Ar "+ str(10.0*positions[i,0]) + " " + str(10.0*positions[i,1]) + " " + str(10.0*positions[i,2]) +"\n")


        # ============ ASSIGNMENT 1.3 ==================================================
        # Implement position and velocity updates of the velocity Verlet integrator.
        # Keep in mind:
        # 1) What should you do, if a particle is outside of the simulation box after 
        #    the position update?
        positions += velocities * dt + 0.5 * accelerations * dt * dt
        positions = apply_periodic_boundary_conditions(positions, box_size)
        accelerations_new = compute_accelerations(positions, masses, epsilon, sigma, box_size, cutoff)
        velocities += 0.5 * (accelerations+accelerations_new) * dt
        accelerations = accelerations_new
    
    return positions, velocities

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

# Example to run the serial simulation with PBC
def main():
 
    N = 100  # Number of particles
    box_size = 5.0  # Size of the simulation box (nm)
    
    positions = np.array(generate_random_points_in_box_3d(box_size, N, 0.4))
    # LJ parameters for Argon
    eps_argon=0.997
    sigma_argon=0.34
    velocities = np.zeros((N, 3))  # Initial velocities
    masses = 40*np.ones(N)  # Masses of the particles

    # ======== ASSIGNMENT 1.4 =====================================
    # Simulations parameters
    # 1) What is a good time step dt?
    # 2) What is a good cutoff (see ASSIGNMENT 1)
    # 3) Initially, the box is 5x5x5 nm^3 containing 100 particles. 
    #    - Perform 1000 simulation steps, look at the result, record
    #      the execution time.
    # 4) Increase the box size to 6,7,8,9,10. Choose the number of 
    #    particles such that it is close to the N/V particle density 
    #    of 3). Perform 1000 simulations steps for each system, and
    #    plot the execution time as a function of N. What do you observe?

    dt =   0.5 # Time step
    steps = 1000  # Number of steps
    cutoff = 1.0
    start_time = time.time()


    # Run the serial Velocity Verlet simulation with PBC
    final_positions, final_velocities = velocity_verlet(
        positions, velocities, masses, dt, steps, box_size,epsilon=eps_argon,sigma=sigma_argon,cutoff=cutoff)

    print("--- %s seconds ---" % (time.time() - start_time))

def plot_execution_times():
    # LJ parameters for Argon
    eps_argon = 0.997
    argon_sigma = 0.34
    dt = 0.0001  # Time step
    steps = 1000  # Number of steps
    #cutoff = 1.25

    base_N = 100
    base_box = 5.0
    desired_density = base_N / (base_box ** 3)

    # Different box sizes to test
    box_sizes = [5, 6, 7, 8, 9, 10]
    N_values = []
    execution_times = []

    for box_size in box_sizes:
        # Calculate N to maintain the same density
        N = int(round(desired_density * (box_size ** 3)))
        # Add Number of particles to list of particle numbers
        N_values.append(N)
        cutoff = box_size*5/12 
        print(f"Running simulation: box_size={box_size}, N={N}")
        
        positions = np.array(generate_random_points_in_box_3d(box_size, N, 0.4))
        velocities = np.zeros((N, 3))
        masses = 40 * np.ones(N)

        start_time = time.time()
        
        final_positions, final_velocities = velocity_verlet(
            positions, velocities, masses, dt, steps, box_size,
            epsilon=eps_argon, sigma=argon_sigma, cutoff=cutoff)
        
        execution_times.append(time.time() - start_time)
        print("--- %s seconds ---" % (time.time() - start_time))
    
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, execution_times, 'bo-', markersize=8, linewidth=2)
    plt.xlabel('Number of particles ', fontsize=12)
    plt.ylabel('Execution time (s)', fontsize=12)
    plt.title('Execution Time vs Number of Particles (100 to 1000)', fontsize=14)
    plt.grid(True)
    
    N_arr = np.array(N_values)
    scale_factor = execution_times[0] / (N_values[0] ** 2)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('execution_time_vs_N.png')
    plt.show()

if __name__ == "__main__":
    plot_execution_times()
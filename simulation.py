import numpy as np
import materials
import geometry
from geometry import geometry
import particle_source
import physics_processes
import matplotlib.pyplot as plt

# --- Simulation Control ---
num_primary_photons = particle_source.xray_source.num_photons
energy_keV_primary, directions_primary, positions_primary = particle_source.xray_source.generate_primary_particles()
step_size_cm = 0.01
energy_threshold_keV = 0.1

# --- Histograms and Counters for Results ---
energy_deposited_in_scintillator_MeV = []
scintillation_photons_reaching_perovskite = [] # Store wavelengths
primary_photons_absorbed_in_perovskite = 0
primary_photons_absorbed_in_shield = 0

# --- Run Simulation ---
for i in range(num_primary_photons):
    current_energy_keV = energy_keV_primary[i]
    current_position = positions_primary[i].copy() # Use a copy to avoid modifying the original
    current_direction = directions_primary[i].copy()

    while current_energy_keV > energy_threshold_keV:
        # Determine current material based on position (simplified 1D along z-axis)
        current_material = None
        distance_travelled_in_current_step = step_size_cm
        interaction_occurred = False

        z_pos = current_position[2]
        if 0 <= z_pos < geometry["shield"].thickness_cm:
            current_material = geometry["shield"].material
        elif geometry["shield"].thickness_cm <= z_pos < (geometry["shield"].thickness_cm + geometry["scintillator"].thickness_cm):
            current_material = geometry["scintillator"].material
        elif (geometry["shield"].thickness_cm + geometry["scintillator"].thickness_cm) <= z_pos < (geometry["shield"].thickness_cm + geometry["scintillator"].thickness_cm + geometry["perovskite"].thickness_cm):
            current_material = geometry["perovskite"].material
        else:
            break # Particle left the defined geometry

        if current_material is not None:
            # Calculate interaction probability
            interaction_prob = physics_processes.calculate_interaction_probability(current_material, current_energy_keV, step_size_cm)

            # Check for interaction
            if np.random.rand() < interaction_prob:
                interaction_occurred = True
                energy_deposited = physics_processes.simulate_photon_interaction(current_energy_keV, current_material)

                if current_material == geometry["scintillator"].material:
                    energy_deposited_MeV = energy_deposited / 1000.0
                    energy_deposited_in_scintillator_MeV.append(energy_deposited_MeV)
                    scintillation_wavelengths = physics_processes.generate_scintillation_photons(energy_deposited_MeV, current_material)
                    # Simplified assumption: all generated scintillation photons reach the perovskite
                    scintillation_photons_reaching_perovskite.extend(scintillation_wavelengths)
                elif current_material == geometry["perovskite"].material:
                    primary_photons_absorbed_in_perovskite += 1
                elif current_material == geometry["shield"].material:
                    primary_photons_absorbed_in_shield += 1
                current_energy_keV = 0 # Primary photon is absorbed

        # Move the photon if no interaction
        if not interaction_occurred:
            current_position += current_direction * distance_travelled_in_current_step

# --- Analyze Results ---
print("--- Primary Photon Interactions ---")
print(f"Number of primary photons absorbed in shield: {primary_photons_absorbed_in_shield}")
print(f"Number of primary photons absorbed in perovskite: {primary_photons_absorbed_in_perovskite}")

print("\n--- Scintillation Light ---")
print(f"Total energy deposited in scintillator (MeV): {np.sum(energy_deposited_in_scintillator_MeV):.4f}")
print(f"Number of scintillation photons (reaching perovskite - simplified): {len(scintillation_photons_reaching_perovskite)}")

# You would then use the 'analysis.py' script to further process and visualize these results

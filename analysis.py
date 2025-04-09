import numpy as np
import matplotlib.pyplot as plt
import simulation  # Import the simulation module to access results

# --- Analysis of Primary Photon Interactions ---
labels = ['Absorbed in Shield', 'Absorbed in Perovskite', 'Escaped Geometry']
sizes = [simulation.primary_photons_absorbed_in_shield,
         simulation.primary_photons_absorbed_in_perovskite,
         simulation.num_primary_photons - simulation.primary_photons_absorbed_in_shield - simulation.primary_photons_absorbed_in_perovskite]
colors = ['grey', 'red', 'lightcoral']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Primary X-ray Photon Interactions')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# --- Analysis of Energy Deposited in Scintillator ---
if simulation.energy_deposited_in_scintillator_MeV:
    plt.figure(figsize=(8, 6))
    plt.hist(simulation.energy_deposited_in_scintillator_MeV, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel("Energy Deposited in Scintillator (MeV)")
    plt.ylabel("Number of Events")
    plt.title("Distribution of Energy Deposited in Scintillator per Primary Photon")
    plt.grid(True)
    plt.show()
else:
    print("No energy deposited in the scintillator to analyze.")

# --- Analysis of Scintillation Photon Wavelengths (if any reached perovskite) ---
if simulation.scintillation_photons_reaching_perovskite:
    plt.figure(figsize=(8, 6))
    plt.hist(simulation.scintillation_photons_reaching_perovskite, bins=np.linspace(360, 400, 21), color='lightgreen', edgecolor='black')
    plt.xlabel("Wavelength of Scintillation Photons (nm)")
    plt.ylabel("Number of Photons")
    plt.title("Wavelength Distribution of Scintillation Photons Reaching Perovskite (Simplified)")
    plt.grid(True)
    plt.show()
else:
    print("No scintillation photons reached the perovskite (based on the simplified model).")

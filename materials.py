import numpy as np
from geometry import geometry

class Material:
    def __init__(self, name, density_g_cm3, elements):
        self.name = name
        self.density_g_cm3 = density_g_cm3
        self.elements = elements  # Dictionary of element: fractional weight
        self.mass_attenuation_coefficients = {} # Energy (keV): cm^2/g
        self.light_yield = 0          # photons/MeV
        self.emission_spectrum = {}   # Wavelength (nm): relative probability
        self.refractive_index = 1.5   # Example value
        self.optical_absorption_length = np.inf # Assume transparent initially
        self.optical_scattering_length = np.inf # Assume no scattering initially

# --- Define Materials ---

# Placeholder for Shielding (Lead-doped CFRP - need approximate elemental composition)
shielding_elements = {"Pb": 0.2, "C": 0.6, "fiber": 0.2} # Example - needs accurate values
shielding = Material("Lead-doped CFRP", 1.7, shielding_elements) # Example density

# Scintillator (PPO + Bis-MSB in LAB - approximate)
scintillator_elements = {"C": 0.88, "H": 0.12} # Very rough approximation for LAB
scintillator = Material("Scintillator (PPO+Bis-MSB/LAB)", 0.87, scintillator_elements)
scintillator.light_yield = 10000 # photons/MeV (example)
scintillator.emission_spectrum = {
    wavelength: np.exp(-((wavelength - 380) / 20)**2) for wavelength in np.linspace(360, 400, 50)
}
scintillator.refractive_index = 1.58 # Approximate for LAB
scintillator.optical_absorption_length = 100.0 # cm (example)
scintillator.optical_scattering_length = 200.0 # cm (example)

# Perovskite (Example - needs accurate elemental composition)
perovskite_elements = {"C": 0.2, "H": 0.2, "N": 0.2, "I": 0.4} # Very rough
perovskite = Material("Perovskite", 4.3, perovskite_elements)
# We'll need QE data later

materials = {
    "shield": shielding,
    "scintillator": scintillator,
    "perovskite": perovskite
}

# Assign materials to layers in geometry
geometry["shield"].material = materials["shield"]
geometry["scintillator"].material = materials["scintillator"]
geometry["perovskite"].material = materials["perovskite"]

import numpy as np

class PhotonSource:
    def __init__(self, energy_keV, num_photons, beam_direction=np.array([0, 0, 1])): # Assuming beam along z-axis
        self.energy_keV = energy_keV
        self.num_photons = num_photons
        self.beam_direction = beam_direction / np.linalg.norm(beam_direction) # Normalize direction

    def generate_primary_particles(self):
        energies = np.full(self.num_photons, self.energy_keV) # Monoenergetic beam
        directions = np.tile(self.beam_direction, (self.num_photons, 1))
        positions = np.zeros((self.num_photons, 3)) # Starting at z=0 (front of the setup)
        return energies, directions, positions

# Define the X-ray source (example: 10 keV monoenergetic beam)
xray_energy_keV = 10.0
num_primary_photons = 10000
xray_source = PhotonSource(xray_energy_keV, num_primary_photons)

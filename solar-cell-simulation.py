import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from matplotlib.lines import Line2D

class SolarCellSimulation:
    def __init__(self):
        # Cell structure dimensions (in micrometers)
        self.dimensions = {
            'radiation_shield': {'thickness': 500, 'position': 0},
            'scintillator': {'thickness': 1000, 'position': 500},
            'perovskite': {'thickness': 500, 'position': 1500},
            'cigs': {'thickness': 1500, 'position': 2000},
            'cnt_heat_spreader': {'thickness': 200, 'position': 3500},
            'silicon': {'thickness': 300000, 'position': 3700},  # 300 micrometers
            'cnt_bottom': {'thickness': 500, 'position': 307000}
        }
        
        # Material properties
        self.materials = {
            'radiation_shield': {
                'color': 'lightgreen',
                'x_ray_attenuation': 0.7,    # Fraction of x-rays attenuated
                'gamma_attenuation': 0.5,     # Fraction of gamma rays attenuated
                'visible_attenuation': 0.1    # Fraction of visible light attenuated
            },
            'scintillator': {
                'color': 'lightblue',
                'x_ray_attenuation': 0.8,      # High absorption for x-rays
                'visible_transmission': 0.9,   # Good transmission of visible light
                'quantum_efficiency': 0.85,    # QE for converting x-rays to visible photons
                'emission_wavelength': 400     # nm (UV-Blue region)
            },
            'perovskite': {
                'color': 'brown',
                'bandgap': 1.6,               # eV
                'absorption_coefficients': {  # Wavelength (nm) -> absorption coefficient (cm^-1)
                    300: 20000, 400: 15000, 500: 10000, 600: 5000, 700: 2000, 800: 1000
                },
                'qd_enhancement': 1.5         # QD enhancement factor for UV light
            },
            'cigs': {
                'color': 'darkgreen',
                'bandgap': 1.1,               # eV
                'absorption_coefficients': {  # Wavelength (nm) -> absorption coefficient (cm^-1)
                    400: 5000, 500: 10000, 600: 15000, 700: 20000, 800: 15000, 900: 10000, 1000: 5000
                }
            },
            'cnt_heat_spreader': {
                'color': 'black',
                'thermal_conductivity': 2000,  # W/mK
                'visible_attenuation': 0.1     # Minimal light attenuation
            },
            'silicon': {
                'color': 'gray',
                'bandgap': 1.1,               # eV
                'absorption_coefficients': {  # Wavelength (nm) -> absorption coefficient (cm^-1)
                    700: 1000, 800: 2000, 900: 5000, 1000: 10000, 1100: 5000, 1200: 1000
                }
            },
            'cnt_bottom': {
                'color': 'black',
                'visible_attenuation': 0.9     # High attenuation (reflective back contact)
            }
        }
        
        # Radiation spectra (simplified)
        self.spectra = {
            'solar': {  # Wavelength (nm) -> relative intensity
                300: 0.4, 400: 0.7, 500: 0.9, 600: 1.0, 700: 0.9, 800: 0.8, 900: 0.7, 
                1000: 0.6, 1100: 0.5, 1200: 0.4
            },
            'x_ray': {  # Energy (keV) -> relative intensity
                5: 0.5, 10: 1.0, 20: 0.8, 30: 0.5, 50: 0.3, 100: 0.1
            },
            'gamma': {  # Energy (keV) -> relative intensity
                200: 0.5, 500: 1.0, 1000: 0.7, 2000: 0.3
            },
            'cosmic': {  # Energy (keV) -> relative intensity
                5000: 0.5, 10000: 0.3, 20000: 0.1
            }
        }
        
        # Simulation results
        self.results = {
            'photon_paths': [],
            'absorbed': {layer: {'count': 0, 'energy': 0.0} for layer in self.dimensions},
            'missed': {'count': 0, 'energy': 0.0},
            'scintillation': {'generated': 0, 'escaped': 0, 'absorbed': 0}
        }
        
        # Constants
        self.h = 4.135667e-15  # Planck's constant in eV·s
        self.c = 299792458     # Speed of light in m/s

    def _wavelength_to_energy(self, wavelength_nm):
        """Convert wavelength in nm to energy in eV"""
        return (self.h * self.c) / (wavelength_nm * 1e-9)
    
    def _energy_to_wavelength(self, energy_ev):
        """Convert energy in eV to wavelength in nm"""
        return (self.h * self.c) / energy_ev * 1e9
    
    def _interpolate(self, x, x_dict):
        """Linearly interpolate a value from a dictionary of points"""
        keys = sorted(x_dict.keys())
        if x <= keys[0]:
            return x_dict[keys[0]]
        if x >= keys[-1]:
            return x_dict[keys[-1]]
        
        # Find bracketing keys
        for i in range(len(keys)-1):
            if keys[i] <= x <= keys[i+1]:
                x1, x2 = keys[i], keys[i+1]
                y1, y2 = x_dict[x1], x_dict[x2]
                return y1 + (y2-y1) * (x-x1)/(x2-x1)
        
        return 0  # Fallback
    
    def _absorption_probability(self, layer, wavelength=None, energy_kev=None):
        """Calculate absorption probability based on material and photon properties"""
        thickness_cm = self.dimensions[layer]['thickness'] * 1e-4  # Convert μm to cm
        
        # Handle high-energy photons (X-rays, gamma rays)
        if energy_kev is not None:
            if layer == 'radiation_shield':
                if energy_kev < 100:  # X-ray range
                    return self.materials[layer]['x_ray_attenuation']
                else:  # Gamma ray range
                    return self.materials[layer]['gamma_attenuation']
            elif layer == 'scintillator':
                return self.materials[layer]['x_ray_attenuation']
            else:
                # Other layers have minimal high-energy absorption
                return 0.05
        
        # Handle optical photons
        if wavelength is not None:
            if layer in ['perovskite', 'cigs', 'silicon']:
                # Get absorption coefficient for this wavelength
                alpha = self._interpolate(wavelength, self.materials[layer]['absorption_coefficients'])
                
                # For perovskite with QD enhancement in UV range
                if layer == 'perovskite' and wavelength < 450:
                    alpha *= self.materials[layer]['qd_enhancement']
                
                # Bandgap cutoff - no absorption above the bandgap wavelength
                bandgap_wavelength = self._energy_to_wavelength(self.materials[layer]['bandgap'])
                if wavelength > bandgap_wavelength:
                    return 0
                
                # Beer-Lambert law
                return 1 - np.exp(-alpha * thickness_cm)
            
            elif layer == 'scintillator':
                # Scintillator is relatively transparent to visible light
                return 1 - self.materials[layer]['visible_transmission']
            
            elif layer in ['radiation_shield', 'cnt_heat_spreader', 'cnt_bottom']:
                return self.materials[layer]['visible_attenuation']
        
        return 0.05  # Default minimal absorption

    def simulate_photon(self, photon_type):
        """Simulate a single photon's path through the solar cell"""
        # Initial photon properties based on type
        if photon_type == 'solar':
            # Sample wavelength from solar spectrum
            wavelengths = list(self.spectra['solar'].keys())
            weights = list(self.spectra['solar'].values())
            wavelength = random.choices(wavelengths, weights=weights)[0]
            energy_ev = self._wavelength_to_energy(wavelength)
            energy_kev = energy_ev / 1000
        elif photon_type == 'x_ray':
            # Sample energy from X-ray spectrum
            energies = list(self.spectra['x_ray'].keys())
            weights = list(self.spectra['x_ray'].values())
            energy_kev = random.choices(energies, weights=weights)[0]
            energy_ev = energy_kev * 1000
            wavelength = None
        elif photon_type == 'gamma':
            # Sample energy from gamma spectrum
            energies = list(self.spectra['gamma'].keys())
            weights = list(self.spectra['gamma'].values())
            energy_kev = random.choices(energies, weights=weights)[0]
            energy_ev = energy_kev * 1000
            wavelength = None
        elif photon_type == 'cosmic':
            # Sample energy from cosmic ray spectrum
            energies = list(self.spectra['cosmic'].keys())
            weights = list(self.spectra['cosmic'].values())
            energy_kev = random.choices(energies, weights=weights)[0]
            energy_ev = energy_kev * 1000
            wavelength = None
        
        # Initial position and path
        position = -100  # Start 100μm before the first layer
        path = [(position, 0)]  # (position, energy)
        
        # Current direction (1 = forward, -1 = backward)
        direction = 1
        
        # Track the photon through the layers
        current_layer = None
        absorbed = False
        
        while position < self.dimensions['cnt_bottom']['position'] + self.dimensions['cnt_bottom']['thickness'] + 100:
            # Determine which layer the photon is in
            for layer, specs in self.dimensions.items():
                if specs['position'] <= position < specs['position'] + specs['thickness']:
                    current_layer = layer
                    break
            else:
                current_layer = None  # In free space
            
            # Step size
            step = 50 if current_layer is None else 10  # Smaller steps in material
            position += step * direction
            
            # Add point to path
            path.append((position, energy_ev))
            
            # Check for absorption
            if current_layer:
                abs_prob = self._absorption_probability(
                    current_layer, wavelength, 
                    energy_kev if photon_type in ['x_ray', 'gamma', 'cosmic'] else None
                )
                
                if random.random() < abs_prob:
                    # Photon absorbed
                    absorbed = True
                    self.results['absorbed'][current_layer]['count'] += 1
                    self.results['absorbed'][current_layer]['energy'] += energy_ev
                    
                    # Handle scintillation process
                    if current_layer == 'scintillator' and photon_type in ['x_ray', 'gamma', 'cosmic']:
                        # Generate optical photons
                        photons_generated = int(energy_ev * self.materials['scintillator']['quantum_efficiency'] / 3)  # ~3eV per visible photon
                        self.results['scintillation']['generated'] += photons_generated
                        
                        # Estimate escape fraction (depends on position within scintillator)
                        relative_depth = (position - self.dimensions['scintillator']['position']) / self.dimensions['scintillator']['thickness']
                        escape_prob = 0.8 * (1 - relative_depth)  # Higher chance to escape if closer to the front
                        
                        # Track escaped photons
                        escaped_photons = int(photons_generated * escape_prob)
                        self.results['scintillation']['escaped'] += escaped_photons
                        
                        # Track how many of these are absorbed in active layers
                        for _ in range(escaped_photons):
                            scint_wavelength = random.normalvariate(
                                self.materials['scintillator']['emission_wavelength'], 20)
                            
                            # Check absorption in perovskite
                            perov_abs_prob = self._absorption_probability('perovskite', scint_wavelength)
                            if random.random() < perov_abs_prob:
                                self.results['scintillation']['absorbed'] += 1
                                self.results['absorbed']['perovskite']['count'] += 1
                                self.results['absorbed']['perovskite']['energy'] += self._wavelength_to_energy(scint_wavelength)
                    
                    break
            
            # Check if we've gone past all layers
            if position > self.dimensions['cnt_bottom']['position'] + self.dimensions['cnt_bottom']['thickness']:
                self.results['missed']['count'] += 1
                self.results['missed']['energy'] += energy_ev
                break
                
            # Check if we've gone back past the start
            if position < -100:
                self.results['missed']['count'] += 1 
                self.results['missed']['energy'] += energy_ev
                break
        
        # Add final path info to results
        self.results['photon_paths'].append({
            'path': path,
            'type': photon_type,
            'absorbed': absorbed,
            'layer': current_layer if absorbed else None,
            'wavelength': wavelength,
            'energy_kev': energy_kev
        })

    def run_simulation(self, num_photons=1000):
        """Run the full simulation with the specified number of photons"""
        # Reset results
        self.results = {
            'photon_paths': [],
            'absorbed': {layer: {'count': 0, 'energy': 0.0} for layer in self.dimensions},
            'missed': {'count': 0, 'energy': 0.0},
            'scintillation': {'generated': 0, 'escaped': 0, 'absorbed': 0}
        }
        
        # Define distribution of photon types
        photon_types = {
            'solar': 0.7,   # 70% solar photons
            'x_ray': 0.15,  # 15% X-rays
            'gamma': 0.1,   # 10% gamma rays
            'cosmic': 0.05  # 5% cosmic rays
        }
        
        # Generate photon counts for each type
        counts = {ptype: int(num_photons * fraction) for ptype, fraction in photon_types.items()}
        
        # Adjust to ensure we have exactly num_photons
        total = sum(counts.values())
        if total < num_photons:
            counts['solar'] += (num_photons - total)
        
        # Run simulation for each photon type
        print(f"Running simulation with {num_photons} photons...")
        for ptype, count in counts.items():
            for _ in range(count):
                self.simulate_photon(ptype)
        
        print(f"Simulation complete.")
        self.print_results()

    def print_results(self):
        """Print simulation results"""
        print("\n=== Absorption Statistics ===")
        total_photons = sum(layer['count'] for layer in self.results['absorbed'].values()) + self.results['missed']['count']
        total_energy = sum(layer['energy'] for layer in self.results['absorbed'].values()) + self.results['missed']['energy']
        
        for layer, stats in self.results['absorbed'].items():
            if stats['count'] > 0:
                photon_percent = (stats['count'] / total_photons) * 100
                energy_percent = (stats['energy'] / total_energy) * 100
                print(f"{layer.capitalize()}: {stats['count']} photons ({photon_percent:.1f}%), "
                      f"{stats['energy']/1000:.2f} keV ({energy_percent:.1f}%)")
        
        if self.results['missed']['count'] > 0:
            photon_percent = (self.results['missed']['count'] / total_photons) * 100
            energy_percent = (self.results['missed']['energy'] / total_energy) * 100
            print(f"Missed: {self.results['missed']['count']} photons ({photon_percent:.1f}%), "
                  f"{self.results['missed']['energy']/1000:.2f} keV ({energy_percent:.1f}%)")
        
        print("\n=== Scintillation Statistics ===")
        scint = self.results['scintillation']
        print(f"Generated scintillation photons: {scint['generated']}")
        if scint['generated'] > 0:
            escape_percent = (scint['escaped'] / scint['generated']) * 100
            print(f"Escaped scintillation photons: {scint['escaped']} ({escape_percent:.1f}%)")
            if scint['escaped'] > 0:
                abs_percent = (scint['absorbed'] / scint['escaped']) * 100
                print(f"Absorbed in active layers: {scint['absorbed']} ({abs_percent:.1f}%)")
                
        # Calculate approximate conversion efficiency
        high_energy_input = sum(p['energy_kev'] * 1000 for p in self.results['photon_paths'] 
                                if p['type'] in ['x_ray', 'gamma', 'cosmic'])
        scint_energy_out = scint['absorbed'] * 3  # Assuming 3eV per absorbed photon
        
        if high_energy_input > 0:
            conv_eff = (scint_energy_out / high_energy_input) * 100
            print(f"\nHigh-energy to electrical conversion efficiency: {conv_eff:.2f}%")

    def visualize_layers(self):
        """Visualize the solar cell structure"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw cell layers
        width = 0.4  # Width of cell in arbitrary units
        
        # Calculate log-scale positions for better visualization (some layers are very thin)
        log_positions = {}
        for layer, specs in self.dimensions.items():
            if layer == list(self.dimensions.keys())[0]:  # First layer
                log_positions[layer] = {'position': specs['position'], 'thickness': specs['thickness']}
            else:
                # Use log scale for layer thickness if very thin compared to others
                thickness = specs['thickness']
                if thickness < 1000:  # If less than 1μm
                    display_thickness = max(500, thickness)  # Minimum display thickness
                else:
                    display_thickness = thickness
                log_positions[layer] = {'position': log_positions[list(self.dimensions.keys())[list(self.dimensions.keys()).index(layer)-1]]['position'] + 
                                                  log_positions[list(self.dimensions.keys())[list(self.dimensions.keys()).index(layer)-1]]['thickness'],
                                      'thickness': display_thickness}
        
        # Draw layers with adjusted thicknesses
        for layer, specs in log_positions.items():
            rect = patches.Rectangle(
                (-width/2, specs['position']), 
                width, specs['thickness'],
                facecolor=self.materials[layer]['color'],
                edgecolor='black',
                alpha=0.7,
                label=layer.capitalize()
            )
            ax.add_patch(rect)
            
            # Add layer name
            ax.text(-width/2 - 0.05, specs['position'] + specs['thickness']/2, 
                   layer.capitalize(), va='center', ha='right')
        
        # Set axis limits
        max_pos = max(specs['position'] + specs['thickness'] for specs in log_positions.values())
        ax.set_xlim(-width - 0.2, width/2 + 0.2)
        ax.set_ylim(-5000, max_pos + 5000)
        
        # Add title and labels
        ax.set_title("Multi-Junction Solar Cell Structure")
        ax.set_xlabel("Width (arbitrary units)")
        ax.set_ylabel("Position (μm)")
        
        plt.tight_layout()
        plt.show()

    def visualize_paths(self, max_paths=100):
        """Visualize photon paths through the cell"""
        if not self.results['photon_paths']:
            print("Run simulation first!")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw cell layers (simplified)
        width = 0.4
        for layer, specs in self.dimensions.items():
            # Use log scale for visualization
            if specs['thickness'] < 1000:
                display_thickness = 2000  # Minimum display thickness for visibility
            else:
                display_thickness = specs['thickness']
            
            rect = patches.Rectangle(
                (-width/2, specs['position']), 
                width, display_thickness,
                facecolor=self.materials[layer]['color'],
                edgecolor='black',
                alpha=0.3
            )
            ax.add_patch(rect)
            
            # Add layer name
            ax.text(-width/2 - 0.05, specs['position'] + display_thickness/2, 
                   layer.capitalize(), va='center', ha='right', fontsize=8)
        
        # Plot photon paths (limited to max_paths for clarity)
        paths_to_show = min(len(self.results['photon_paths']), max_paths)
        selected_paths = random.sample(self.results['photon_paths'], paths_to_show)
        
        # Colors for different photon types
        colors = {
            'solar': 'gold',
            'x_ray': 'red',
            'gamma': 'purple',
            'cosmic': 'blue',
            'scintillation': 'cyan'
        }
        
        # Plot paths
        for photon in selected_paths:
            x_vals = [0] * len(photon['path'])  # Center all paths
            y_vals = [pos for pos, _ in photon['path']]
            
            ax.plot(x_vals, y_vals, color=colors[photon['type']], 
                   alpha=0.5, linewidth=1, marker='', linestyle='-')
        
        # Create legend
        legend_elements = [Line2D([0], [0], color=color, lw=2, label=ptype.capitalize())
                          for ptype, color in colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set axis limits
        max_pos = max(specs['position'] + specs['thickness'] for specs in self.dimensions.values())
        ax.set_xlim(-width - 0.2, width/2 + 0.2)
        ax.set_ylim(-10000, max_pos + 10000)
        
        # Add title and labels
        ax.set_title(f"Photon Paths (showing {paths_to_show} of {len(self.results['photon_paths'])} photons)")
        ax.set_xlabel("Width (arbitrary units)")
        ax.set_ylabel("Position (μm)")
        
        plt.tight_layout()
        plt.show()

    def plot_absorption_stats(self):
        """Plot absorption statistics by layer"""
        if not self.results['photon_paths']:
            print("Run simulation first!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Prepare data
        layers = list(self.results['absorbed'].keys())
        photon_counts = [self.results['absorbed'][layer]['count'] for layer in layers] + [self.results['missed']['count']]
        energy_values = [self.results['absorbed'][layer]['energy']/1000 for layer in layers] + [self.results['missed']['energy']/1000]
        labels = [layer.capitalize() for layer in layers] + ['Missed']
        
        # Colors matching the materials
        colors = [self.materials[layer]['color'] for layer in layers] + ['silver']
        
        # Photon count plot
        ax1.bar(range(len(photon_counts)), photon_counts, color=colors)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_title('Photon Absorption by Layer')
        ax1.set_ylabel('Number of Photons')
        
        # Energy absorption plot
        ax2.bar(range(len(energy_values)), energy_values, color=colors)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_title('Energy Absorption by Layer')
        ax2.set_ylabel('Energy (keV)')
        
        plt.tight_layout()
        plt.show()

    def plot_spectral_response(self):
        """Plot the spectral response of the cell"""
        # Define wavelength range to test
        wavelengths = np.linspace(300, 1200, 100)
        
        # Calculate absorption in each active layer
        perovskite_abs = [self._absorption_probability('perovskite', wl) for wl in wavelengths]
        cigs_abs = [self._absorption_probability('cigs', wl) for wl in wavelengths]
        silicon_abs = [self._absorption_probability('silicon', wl) for wl in wavelengths]
        
        # Plot responses
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wavelengths, perovskite_abs, label='Perovskite', color=self.materials['perovskite']['color'])
        ax.plot(wavelengths, cigs_abs, label='CIGS', color=self.materials['cigs']['color'])
        ax.plot(wavelengths, silicon_abs, label='Silicon', color=self.materials['silicon']['color'])
        
        # Add the solar spectrum for reference
        solar_x = list(self.spectra['solar'].keys())
        solar_y = list(self.spectra['solar'].values())
        ax2 = ax.twinx()
        ax2.plot(solar_x, solar_y, 'k--', label='Solar Spectrum', alpha=0.5)
        
        # Labels and title
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Absorption Probability')
        ax2.set_ylabel('Relative Intensity')
        ax.set_title('Layer Absorption vs. Wavelength')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.tight_layout()
        plt.show()


# Example usage of the simulation
if __name__ == "__main__":
    # Create simulation instance
    sim = SolarCellSimulation()
    
    # Run simulation with 5000 photons
    sim.run_simulation(5000)
    
    # Visualize the cell structure
    sim.visualize_layers()
    
    # Show photon paths
    sim.visualize_paths(max_paths=200)
    
    # Plot absorption statistics
    sim.plot_absorption_stats()
    
    # Plot spectral response
    sim.plot_spectral_response()

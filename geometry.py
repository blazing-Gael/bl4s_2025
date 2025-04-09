class Layer:
    def __init__(self, name, material, thickness_mm, length_mm, breadth_mm):
        self.name = name
        self.material = material
        self.thickness_cm = thickness_mm / 10.0  # Convert mm to cm
        self.length_cm = length_mm / 10.0
        self.breadth_cm = breadth_mm / 10.0

# Define layer dimensions (example values - you need to determine realistic ones)
shield_thickness_mm = 2.0
scintillator_thickness_mm = 5.0
perovskite_thickness_mm = 0.001  # 1 micrometer

lateral_length_mm = 10.0
lateral_breadth_mm = 10.0

# Define the layers
shield_layer = Layer("Shield", None, shield_thickness_mm, lateral_length_mm, lateral_breadth_mm)
scintillator_layer = Layer("Scintillator", None, scintillator_thickness_mm, lateral_length_mm, lateral_breadth_mm)
perovskite_layer = Layer("Perovskite", None, perovskite_thickness_mm, lateral_length_mm, lateral_breadth_mm)

geometry = {
    "shield": shield_layer,
    "scintillator": scintillator_layer,
    "perovskite": perovskite_layer
}

import math

class PayloadBayStructure():
    def __init__(self, payloadHeight, payloadRaidus, lowerSategeRadius, payloadMass, S_lower_rocket,verbose=True):
        self.payloadHeight = payloadHeight
        self.payloadRaidus = payloadRaidus
        self.lowerSategeRadius = lowerSategeRadius
        self.payloadMass = payloadMass
        self.S_lower_rocket = S_lower_rocket # Rocket total surface area - Precisa integrar isso com os outros pontos
        self.verbose = verbose


        self.totalHeight = 20/9 * self.payloadHeight
        self.triangleHeight = 7/20 * self.totalHeight
        self.cylinderHeight = 9/20 * self.totalHeight
        self.frustrumHeight = 4/20 * self.totalHeight


        #self.triangleHeight = 7/20 * self.totalHeight
        #self.cylinderHeight = 9/20 * self.totalHeight
        #self.frustrumHeight = 4/20 * self.totalHeight


    def estimate_surface_area(self):
        self.coneArea = math.pi * self.triangleHeight * self.payloadRaidus
        self.cylinderArea = 2 * math.pi * self.cylinderHeight * self.payloadRaidus
        self.frustrumArea = math.pi * (self.payloadRaidus + self.lowerSategeRadius) * math.sqrt( ( (self.payloadRaidus - self.lowerSategeRadius) ** 2  ) + (self.frustrumHeight ** 2) )
        if self.verbose:
            print(f"Cone Area: {self.coneArea} [kg]")
            print(f"Cylinder Area: {self.cylinderArea} [kg]")
            print(f"Frustrum Area: {self.frustrumArea} [kg]")

    def get_mass_from_area(self, area):
        mass = 4.95 * (area ** 1.15)
        return mass


    def estimate_fairing_mass(self):
        self.coneMass = self.get_mass_from_area(self.coneArea)
        self.cylinderMass = self.get_mass_from_area(self.cylinderArea)
        self.frustrumMass = self.get_mass_from_area(self.frustrumArea)
        self.totalPayloadFairingMass = self.coneMass +  self.cylinderMass + self.frustrumMass
        self.S_rocket = self.S_lower_rocket + self.totalPayloadFairingMass
        if self.verbose:
            print(f"Total Payload Fairing Mass: {self.totalPayloadFairingMass} [kg]")

    def estimate_payload_bay_adapter(self):
        self.payloadAdapterMass = 0.0477536 * (self.payloadMass ** (1.01317))
        if self.verbose:
            print(f"Payload Adapter Mass: {self.payloadAdapterMass} [kg]")

    def estimate_avionics_mass(self):
        K_rl = 0.7
        S_rocket = self.S_rocket
        TRF_electronic = 0.75
        TRF_power = 0.18
        self.electronicMass = K_rl * (246.76 + 1.3183 * S_rocket) * (1 - TRF_electronic)
        self.powerMass = K_rl * 0.405 * self.electronicMass * (1 - TRF_power)

        if self.verbose:
            print(f"Electronic Mass: {self.electronicMass} [kg]")
            print(f"Power Mass: {self.powerMass} [kg]")
        return

    def estimate_total_mass(self):
        self.totalMass = self.totalPayloadFairingMass + self.electronicMass + self.powerMass + self.payloadMass
        if self.verbose:
            print(f"Total Payload Mass: {self.totalMass} [kg]")
        return

    def estimate_all(self):
        self.estimate_surface_area()
        self.estimate_fairing_mass()
        self.estimate_avionics_mass()
        self.estimate_total_mass()
    
    def print_all_parameters(self):
        print(f"Cone Area: {self.coneArea} [kg]")    
        print(f"Cylinder Area: {self.cylinderArea} [kg]")
        print(f"Frustrum Area: {self.frustrumArea} [kg]")
        print(f"Total Payload Fairing Mass: {self.totalPayloadFairingMass} [kg]")
        #print(f"Payload Adapter Mass: {self.payloadAdapterMass} [kg]")
        print(f"Electronic Mass: {self.electronicMass} [kg]")
        print(f"Power Mass: {self.powerMass} [kg]")
        print(f"Total Payload Mass: {self.totalMass} [kg]")


if __name__ == "__main__":
    payload = PayloadBayStructure(6.7, 4.6/2, 1.83, 5000, 1000)
    payload.estimate_all()
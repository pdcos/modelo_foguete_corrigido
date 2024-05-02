import math
import numpy as np
import pandas as pd

fuelDensityDict = {"LOX": 1140, "LH2": 71, "RP1": 820, "N2O4": 1450, "UDMH": 793}

tankRegressionList = [{"x": 12.158, "y": 0.9328, "chemical_compound": ["LOX", "RP1", "N2O4", "UDMH"]},
                      {"x": 9.0911, "y": 0.9896, "chemical_compound": ["LH2"]}]


class PressurizationTank():
    def __init__(self,
                 tankPressure,
                 tankVolume,
                 R_gas,
                 T_gas,
                 gamma_gas,
                 gasPressure,
                 s_m = 1.5,
                 t_min = 2,
                 alloyDensity = 2590,
                 maximumStress = 483,
                 A_profile = 8*1e-4,
                 n_profile = 3,
                 l_profile = 1,
                 verbose:bool = False):

        self.tankPressure = tankPressure
        self.tankVolume = tankVolume
        self.R_gas = R_gas
        self.T_gas = T_gas
        self.gamma_gas = gamma_gas
        self.gasPressure = gasPressure
        self.verbose = verbose
        self.alloyDensity = alloyDensity
        self.maximumStress = maximumStress
        self.A_profile = A_profile
        self.n_profile = n_profile
        self.l_profile = l_profile
        self.s_m = s_m
        self.t_min = t_min

    def estimate_gas_mass(self):
        num =  1.1 * self.tankPressure * self.tankVolume * self.gamma_gas
        den = self.R_gas * self.T_gas * (1 - (self.tankPressure/(self.gasPressure * 1e6)))
        self.gasMass = num/den
        if self.verbose:
            print(f"Pressurization Gas Mass: {self.gasMass} [kg]")
        return

    def estimate_gas_volume(self):
        #PV = mRT (R em J/(kg * K))
        self.gasVolume = self.gasMass * self.R_gas * self.T_gas / (self.gasPressure * 1e6)
        if self.verbose:
            print(f"Pressurization Gas Volume: {self.gasVolume} [m3]")
        return

    def estimate_tank_radius(self):
        # Supondo um tanque esferico
        # v = 4/3 pi r^3
        self.tankRadius = ((3/(4 * math.pi))) ** (1/3)
        if self.verbose:
            print(f"Pressurization Tank Radius: {self.tankRadius} [m]")
        return

    def estimate_tank_surface(self):
        self.tankSurface = 4 * math.pi * (self.tankRadius ** 2)
        if self.verbose:
            print(f"Pressurization Tank Surface Area: {self.tankSurface} [m2]")

        return
    
    def estimate_tank_thickness(self): 
        self.tankThickness = self.s_m * self.gasPressure * self.tankRadius / (2 * self.maximumStress)
        if self.verbose:
            print(f"Pressurization Tank Surface Thickness: {self.tankThickness * 1000} [mm]")

    def estimate_tank_mass(self):
        self.tankMass = self.alloyDensity * self.tankSurface * (self.tankThickness + self.A_profile * self. n_profile * self.l_profile) 
        if self.verbose:
            print(f"Pressurization Tank Mass: {self.tankMass} [kg]")

    def estimate_all(self):
        self.estimate_gas_mass()
        self.estimate_gas_volume()
        self.estimate_tank_radius()
        self.estimate_tank_thickness()
        self.estimate_tank_surface()
        self.estimate_tank_mass()
        return


class interStageStructure():
    def __init__(self,
                 propellantMass,
                 oxName,
                 fuelName,
                 MR,
                 radius,
                 maxEngineThrust,
                 upperMass,
                 engineLength = 2.92,
                 lowerRadius = False,
                 length = None,
                 f_ull = 1.05,
                 s_m = 1.5,
                 t_min = 2,
                 maximumStress = 483,
                 alloyDensity = 2590,
                 tankPressure = 0.1, # MPa
                 A_profile = 8*1e-4,
                 n_profile = 3,
                 l_profile = 1,
                 f_ins = 2.88,
                 verbose:bool=True):

        self.propellantMass = propellantMass
        self.oxName = oxName
        self.fuelName = fuelName
        self.MR = MR
        self.length = length
        self.radius = radius
        self.f_ull = f_ull # Fator aplicato para permitir expansao de gases. Por padrão é 5%
        self.s_m = s_m # Safety margin - largura da parede
        self.t_min = t_min # Minimum thickness (mm)
        self.maximumStress = maximumStress # Maximum allowable stress (MPa)
        self.alloyDensity = alloyDensity # Densidade do metal utilizado nas estruturas
        self.tankPressure = tankPressure # Pressao nos tanques (MPa)
        self.A_profile = A_profile # Area de perfil dos stringers com 4mm de grossura, 1m entre anies e 0.5 entre stringers (m^2)
        self.n_profile = n_profile # Numero de perfis por metro quadrado, cada um com um comprimento de 1m 
        self.l_profile = l_profile # Comprimento dos perfis (m)
        self.f_ins = f_ins # Fator de isolamento que depente do combustivel (Oxigenio = 2.88 kg/m^2)
        self.verbose = verbose
        self.maxEngineThrust = maxEngineThrust
        self.lowerRadius = lowerRadius
        self.engineLength = engineLength
        self.upperMass = upperMass

        self.oxTankVolume = None
        self.fuelTankVolume = None
        self.oxTankCylHeight = None
        self.fuelTankCylHeight = None

        self.oxMass = MR * propellantMass / (1 + MR)
        self.fuelMass = propellantMass / (1 + MR)
        if verbose:
            print(f"Ox Mass: {self.oxMass} [kg]")
            print(f"Fuel Mass: {self.fuelMass} [kg]")

    def estimate_tank_volume(self):
        self.oxTankVolume = self.oxMass / fuelDensityDict[self.oxName] * self.f_ull
        self.fuelTankVolume = self.fuelMass / fuelDensityDict[self.fuelName] * self.f_ull
        if self.verbose:
            print(f"Ox Tank Volume: {self.oxTankVolume} [m3]")
            print(f"Fuel Tank Volume: {self.fuelTankVolume} [m3]")
        return
    
    def estimate_lids_volume(self):
        """
        Estima o volume das pontas esféricas do cilindro.
        O raio das esferas é igual ao raio do estágio.
        O volume é o mesmo para ox e fuel porque só depende do raio
        """
        self.lidsVolume = (4/3) * math.pi * (self.radius ** 3)
        if self.verbose:
            print(f"Espherical Lids Volume: {self.lidsVolume} [m3]")
        return
    
    def estimate_cylyndrical_height(self):
        self.oxTankCylHeight = (1/( math.pi * (self.radius ** 2))) * (self.oxTankVolume - self.lidsVolume)
        self.fuelTankCylHeight = (1/( math.pi * (self.radius ** 2))) * (self.fuelTankVolume - self.lidsVolume)
        if self.verbose:
            print(f"Ox Tank Cylyndrical Height: {self.oxTankCylHeight} [m]")
            print(f"Fuel Tank Cylyndrical Height: {self.fuelTankCylHeight} [m]")
        return
    
    def estimate_tank_surface_area(self):
        self.lidsArea = 4 * math.pi * (self.radius ** 2)
        self.oxTankSurface = 2 * math.pi * self.radius * self.oxTankCylHeight + self.lidsArea
        self.fuelTankSurface = 2 * math.pi * self.radius * self.fuelTankCylHeight + self.lidsArea
        if self.verbose:
            print(f"Ox Tank Surface: {self.oxTankSurface} [m2]")
            print(f"Fuel Tank Surface: {self.fuelTankSurface} [m2]")
        return

    def estimate_wall_thickness(self):  
        #Multiplica por 10 para converter
        self.tankThickness = self.s_m * self.tankPressure * self.radius / self.maximumStress
        if (self.tankThickness * 1000) < (self.t_min):
            self.tankThickness = self.t_min/1000 
        if self.verbose:
            print(f"Tank Thickness: {self.tankThickness * 1000} [mm]")
        return
    
    def estimate_tank_mass(self):
        self.oxTankMass = self.alloyDensity * self.oxTankSurface * (self.tankThickness + self.A_profile * self. n_profile * self.l_profile) 
        self.fuelTankMass = self.alloyDensity * self.fuelTankSurface * (self.tankThickness + self.A_profile * self. n_profile * self.l_profile) 
        if self.verbose:
            print(f"Ox Tank Mass: {self.oxTankMass} [kg]")
            print(f"Fuel Tank Mass: {self.fuelTankMass} [kg]")
        return

    def estimate_insulation_mass(self):
        insulationFactor = {"LH2": 2.88, "LOX": 1.123, "RP1": 0}
        self.oxTankInsulationMass = insulationFactor[self.oxName] * self.oxTankSurface
        self.fuelTankInsulationMass = insulationFactor[self.fuelName] * self.fuelTankSurface
        if self.verbose:
            print(f"Ox Tank Insulation Mass: {self.oxTankInsulationMass} [kg]")
            print(f"Fuel Tank Insulation Mass: {self.fuelTankInsulationMass} [kg]")
        return

    def estimate_pressurization_tank_mass(self):
        tankPressure = self.tankPressure * 1e6
        tankVolume = self.oxTankVolume + self.fuelTankVolume
        self.pressurizationTank = PressurizationTank(tankPressure=tankPressure,
                                                tankVolume=tankVolume,
                                                R_gas=2077,
                                                T_gas=293,
                                                gamma_gas=1.667,
                                                gasPressure=29.6,
                                                verbose=False
                                                )
        self.pressurizationTank.estimate_all()
        self.pressurizationTankMass = self.pressurizationTank.tankMass + self.pressurizationTank.gasMass

    def get_mass_from_area(self, area):
        if area < 0:
            return float('nan')
        else:
            mass = 4.95 * (area ** 1.15)
        return mass

    
    def estimate_intertank_structure_mass(self):
        self.intertankStructureArea = 8 * math.pi * (self.radius ** 2)
        self.intertankStructureMass = self.get_mass_from_area(self.intertankStructureArea)
        if self.verbose:
            print(f"Intertank Structure Mass: {self.intertankStructureMass} [kg]")
        return

    def estimate_thrust_frame(self):
        self.thrustFrameMass = (2.04 * 1e-5) * (self.maxEngineThrust ** (1.15))
        if self.verbose:
            print(f"Thrust Frame Mass: {self.thrustFrameMass} [kg]")
        return
    
    def estimate_interstage_structure_mass(self):
        if self.lowerRadius:
            root =  (((self.engineLength +  0.4 * self.radius) ** 2 + (self.radius - self.lowerRadius) ** 2)**(1/2))
            self.interstageSurfaceArea = math.pi * (self.radius + self.lowerRadius) * root
            self.interstageMass = self.get_mass_from_area(self.interstageSurfaceArea)
        else:
            self.interstageMass = 0
        if self.verbose:
            print(f"Interstage Structure Mass: {self.interstageMass} [kg]")
        return
    
    def estimate_separation_explosives_mass(self):
        self.separationExplosivesMass = (8.7 * 1e-4) * self.upperMass
        if self.verbose:
            print(f"Separation Explosives Mass: {self.separationExplosivesMass} [kg]")
        return
    
    def estimate_tank_fairing_mass(self):
        self.oxTankFairingSurfaceArea = 2 * math.pi * (self.oxTankCylHeight + self.radius * 2)
        self.fuelTankFairingSurfaceArea = 2 * math.pi * (self.fuelTankCylHeight + self.radius * 2)
        self.pressurizationTankFairingSurfaceArea = 2 * math.pi * (self.pressurizationTank.tankRadius)
        self.oxTankFairingMass = self.get_mass_from_area(self.oxTankFairingSurfaceArea)
        self.fuelTankFairingMass = self.get_mass_from_area(self.fuelTankFairingSurfaceArea)
        self.pressurizationTankFairingMass = self.get_mass_from_area(self.pressurizationTankFairingSurfaceArea)
        self.tankFairingMass = self.oxTankFairingMass + self.fuelTankFairingMass #+ self.pressurizationTankFairingMass
        if self.verbose:
            print(f"Tank Fairing Mass: {self.tankFairingMass} [kg]")
        return

    def get_total_height(self):
        self.totalHeight = self.oxTankCylHeight + self.fuelTankCylHeight \
                           + 4 * self.radius + \
                            self.engineLength + \
                            2 * self.pressurizationTank.tankRadius 

        if self.verbose:
            print(f"Total Stage Height: {self.totalHeight} [m]")
        return

    


    def get_total_stage_dry_mass(self):
        self.totalMass = self.oxTankMass + self.oxTankInsulationMass \
                        + self.fuelTankMass + self.oxTankInsulationMass \
                        + self.pressurizationTankMass \
                        + self.thrustFrameMass \
                        + self.interstageMass \
                        + self.separationExplosivesMass \
                        + self.tankFairingMass        
                        #+ self.intertankStructureMass \ # Não precisa porque as esferas já estão inclusas em estimate_tank_fairing_mass


        if self.verbose:
            print(f"Total Stage Dry Mass: {self.totalMass} [kg]")
        return

    def get_total_surface_area(self):
        self.totalSurfaceArea = self.oxTankFairingSurfaceArea + \
                                self.fuelTankFairingSurfaceArea + \
                                self.intertankStructureArea + \
                                self.pressurizationTankFairingSurfaceArea
        if self.verbose:
            print(f"Total Stage Surface Area: {self.totalSurfaceArea} [m2]")
        return

    def print_all_parameters(self):
        print(f"Ox Tank Cylyndrical Height: {self.oxTankCylHeight} [m]")
        print(f"Fuel Tank Cylyndrical Height: {self.fuelTankCylHeight} [m]")
        print(f"Ox Tank Volume: {self.oxTankVolume} [m3]")
        print(f"Fuel Tank Volume: {self.fuelTankVolume} [m3]")
        print(f"Intertank Structure Mass: {self.intertankStructureMass} [kg]")
        print(f"Separation Explosives Mass: {self.separationExplosivesMass} [kg]")
        print(f"Tank Fairing Mass: {self.tankFairingMass} [kg]")
        print(f"Total Stage Height: {self.totalHeight} [m]")
        print(f"Total Stage Dry Mass: {self.totalMass} [kg]")
        print(f"Total Stage Surface Area: {self.totalSurfaceArea} [m2]")

    def estimate_all(self):
        self.estimate_tank_volume()
        self.estimate_lids_volume()
        self.estimate_cylyndrical_height()
        self.estimate_tank_surface_area()
        self.estimate_wall_thickness()
        self.estimate_tank_mass()
        self.estimate_insulation_mass()
        self.estimate_pressurization_tank_mass()
        self.estimate_intertank_structure_mass()
        self.estimate_thrust_frame()
        self.estimate_interstage_structure_mass()
        self.estimate_separation_explosives_mass()
        self.estimate_tank_fairing_mass()
        self.get_total_stage_dry_mass()
        self.get_total_height()
        self.get_total_surface_area()
        return
    



if __name__ == "__main__":
    
    # Estimando um upperstage
    #stage = interStageStructure(oxName="LOX",
    #                            fuelName="RP1",
    #                            propellantMass=156883.5,
    #                            MR=2.9,
    #                            radius=2,
    #                            tankPressure=0.1,
    #                            maxEngineThrust=1604988.3,
    #                            lowerRadius=2.6,
    #                            upperMass=10000,
    #                            )
    #stage.estimate_all()
    stage = interStageStructure(oxName="LOX",
                                fuelName="RP1",
                                propellantMass=156883.5,
                                MR=2.9,
                                radius=2,
                                tankPressure=0.1,
                                maxEngineThrust=1 * 1604 * 1000,
                                lowerRadius=False,
                                upperMass=14000,
                                )
    stage.estimate_all()

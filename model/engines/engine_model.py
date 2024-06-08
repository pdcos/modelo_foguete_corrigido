import math
from rocketcea.cea_obj_w_units import CEA_Obj
from proptools import nozzle  
import numpy as np
import joblib
import time
import joblib


class EngineProp:
    def __init__(self,
                MR:float,
                Pc:float,
                eps:float,
                nozzleDiam = None,
                At=None,
                verbose=True,
                reg_model=False, 
                cea_obj=False,
                bound_values=False,
                oxName='LOX',
                fuelName='RP-1'):

        self.MR = MR
        self.Pc = Pc
        self.eps = eps
        self.nozzleDiam = nozzleDiam
        self.At = At
        self.verbose = verbose
        self.reg_model = reg_model
        self.ceaObj= cea_obj
        self.fuelName = fuelName
        self.oxName = oxName


        if bound_values == False:
            self.bound_values = np.array([[0.1e6, 12e6], [1.5, 3.5], [2, 200]])
        else:
            self.bound_values = bound_values
        # if bound_values_cea is None:
        #     self.bound_values_cea = np.array([[0.1e6, 12e6], [1.5, 3.5], [2, 200]])
        # else:
        #     self.bound_values_cea = bound_values_cea
        self.min_mat = self.bound_values.T[0, :]
        self.max_mat = self.bound_values.T[1,:]


        

        # Podemos escolher informar o diametro do throat ou a area
        if ((self.At == None) and (self.nozzleDiam != None)):
            self.At = np.pi * ((self.nozzleDiam/2) ** 2)
        elif ((self.At == None) and (self.nozzleDiam == None)):
            raise("Informar At ou nozzleDiam")
        elif ((self.At != None) and (self.nozzleDiam != None)):
            raise("Informar apenas 1: At ou nozzleDiam")
        
    def calcEngineProperties(self):
        if self.reg_model:
            input = np.array([[self.Pc, self.MR, self.eps]])
            input = (input - self.min_mat)/ (self.max_mat - self.min_mat)

            # input = input * (self.bound_values[:, 1] - self.bound_values[:, 0]) + self.bound_values[:, 0]
            # input = (input - self.bound_values_cea.T[0, :]) / (self.bound_values_cea.T[1, :] - self.bound_values_cea.T[0, :])

            IspSea, IspVac, Cstar, mw, Tc, gamma = self.reg_model.predict(input)[0]
            m_molar = mw/1000
        else:
            IspVac, Cstar, Tc, mw, gamma = self.ceaObj.get_IvacCstrTc_ChmMwGam(Pc=self.Pc, MR=self.MR, eps=self.eps)
            m_molar = mw/1000
            IspSea = self.ceaObj.estimate_Ambient_Isp(Pc=self.Pc, MR=self.MR, eps=self.eps, Pamb=1e5)[0]
        Pc = self.Pc 


        Pe = Pc * nozzle.pressure_from_er(self.eps, gamma)
        # Empuxo no vacuo (N)
        thrustVac = nozzle.thrust(A_t = self.At,
                                  p_c = Pc,
                                  p_e = Pe,
                                  p_a = 0,
                                  gamma=gamma,
                                  er = self.eps)
        # Empuxo no nivel do mar (N)
        thrustSea = nozzle.thrust(A_t = self.At,
                                  p_c = Pc,
                                  p_e = Pe,
                                  p_a = 1e5,
                                  gamma=gamma,
                                  er = self.eps)
        # Fluxo de mass (kg/s)
        massFlow = nozzle.mass_flow(A_t = self.At,
                                     p_c = Pc,
                                     T_c = Tc,
                                     gamma = gamma,
                                     m_molar = m_molar
                                     ) 

        if self.verbose:
            print("Isp Vac (s): " + str(IspVac))
            print("Isp Sea (s): " + str(IspSea))
            print("Mass flow (kg/s): " + str(massFlow))
            print("Thrust Vac (kN): " + str(thrustVac/1000))
            print("Thrust Sea (kN): " + str(thrustSea/1000))

        self.IspVac = IspVac
        self.IspSea = IspSea
        self.massFlow = massFlow
        self.thrustVac = thrustVac
        self.thrustSea = thrustSea
        
    def estimate_engine_mass(self):
        """
        Estima a massa do motor
        
        :param proppelantType: Tipo de propelente - pode ser "Cryogenic-Cryogenic" ou "Cryogenic-Storable"
        :param thrustVac: Empuxo do motor no váculo em Newtons
        :return massTvc: Massa estimada do TVC em kg 
        """
        if self.fuelName == "RP-1":
            propellantType = "Cryogenic-Storable"
        else:
            propellantType = "Cryogenic-Cryogenic"

        if propellantType == "Cryogenic-Cryogenic":
            self.engineMass = 7.54354 * (1e-3) * (self.thrustVac ** (8.85635 * (1e-1))) + 2.02881 * (1e1)
        elif propellantType == "Cryogenic-Storable":
            self.engineMass = 3.75407 * (1e3) * (self.thrustVac ** (7.05627 * (1e-2))) - 8.84790 * (1e3)
        else:
            raise Exception("Selecione um tipo de propelente válido!")
        if self.verbose:
            print(f"Engine Mass: {self.engineMass} [kg]")
        return self.engineMass

    def estimate_tvc_mass(self):
        """
        Estima a massa do Thrust Vector Control System (TVC)
        
        :param thrustVac: Empuxo do motor no váculo em Newtons
        :return massTvc: Massa estimada do TVC em kg 
        """
        self.massTvc = 0.1078 * (self.thrustVac/1e3) + 43.702
        if self.verbose:
            print(f"Engine Mass: {self.massTvc} [kg]")
        return self.massTvc

    def get_total_mass(self):
        self.totalMass = self.massTvc + self.engineMass

    def estimate_all(self):
        self.calcEngineProperties()
        self.estimate_engine_mass()
        self.estimate_tvc_mass()
        self.get_total_mass()
        return
    
    def print_all_parameters(self):
        print("Isp Vac (s): " + str(self.IspVac))
        print("Isp Sea (s): " + str(self.IspSea))
        print("Mass flow (kg/s): " + str(self.massFlow))
        print("Thrust Vac (kN): " + str(self.thrustVac/1000))
        print("Thrust Sea (kN): " + str(self.thrustSea/1000))

    
if __name__ == '__main__':
    cea_obj = ceaObj = CEA_Obj( oxName='LOX', fuelName='RP-1', pressure_units='Pa', cstar_units='m/s', temperature_units='K')
    reg_path = '/home/ubuntu/Mestrado/modelo_foguete_corrigido/improve_exec_speed/data/DecisionTreeRegressor_score_1.0.joblib'
    reg_model = joblib.load(reg_path)

    engine = EngineProp(MR=2.36, 
                        Pc=9.72e6, 
                        eps=117, 
                        nozzleDiam=0.23125,
                        verbose=True,
                        reg_model=reg_model, 
                        cea_obj=cea_obj, 
                        bound_values=False)
    engine.estimate_all()

    print("#"*10)

    bounds = np.array([[7e6, 12e6], [1.5, 2.5], [0.2, 0.3], [30, 200]])
    inputs = np.array([[9.72e6, 2.36, 0.23125, 117]])

    # normalized = (inputs - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    # print(normalized)

    # engine = EngineProp(MR=normalized[0][1], 
    #                     Pc=normalized[0][0], 
    #                     eps=normalized[0][3], 
    #                     nozzleDiam=normalized[0][2],
    #                     verbose=True,
    #                     reg_model=reg_model, 
    #                     cea_obj=cea_obj, 
    #                     bound_values=bounds[[0, 1, 3]])
    # engine.estimate_all()
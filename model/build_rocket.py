import numpy as np
import math
try:
    from model.structure.payload_bay_structure import PayloadBayStructure
    from model.structure.stage_structure import interStageStructure
    from model.engines.engine_model import EngineProp
except:
    from structure.payload_bay_structure import PayloadBayStructure
    from structure.stage_structure import interStageStructure
    from engines.engine_model import EngineProp


class RocketModel():
    def __init__(self, upperEngineParams, firstEngineParams, payloadBayParams, upperStageStructureParams, firstStageStructureParams, deltaV_upperStage, deltaV_landing, deltaV_firstStage, nEnginesUpperStage, nEnignesFirstStage, reg_model=False, cea_obj=False, bound_values=False):
        self.upperEngineParams = upperEngineParams
        self.firstEngineParams = firstEngineParams
        self.payloadBayParams = payloadBayParams
        self.upperStageStructureParams = upperStageStructureParams
        self.firstStageStructureParams = firstStageStructureParams
        self.g0 = 9.81
        self.deltaV_upperStage = deltaV_upperStage
        self.upperStageEngine = None,
        self.deltaV_landing = deltaV_landing
        self.deltaV_firstStage = deltaV_firstStage
        self.nEnginesUpperStage = nEnginesUpperStage
        self.nEnginesFirstStage = nEnignesFirstStage
        self.reg_model = reg_model
        self.cea_obj = cea_obj
        self.bound_values = bound_values
    

    def build_engines(self):
        self.upperStageEngine = EngineProp(#fuelName = self.upperEngineParams["fuelName"],
                                           #oxName = self.upperEngineParams["oxName"],
                                           Pc = self.upperEngineParams["combPressure"],
                                           MR = self.upperEngineParams["MR"],
                                           nozzleDiam = self.upperEngineParams["nozzleDiam"],
                                           eps = self.upperEngineParams["eps"],
                                           verbose=False,
                                           reg_model=self.reg_model,
                                           cea_obj=self.cea_obj,
                                           )
        self.firstStageEngine = EngineProp(#fuelName = self.firstEngineParams["fuelName"],
                                           #oxName = self.firstEngineParams["oxName"],
                                           Pc = self.firstEngineParams["combPressure"],
                                           MR = self.firstEngineParams["MR"],
                                           nozzleDiam = self.firstEngineParams["nozzleDiam"],
                                           eps = self.firstEngineParams["eps"],
                                           verbose=False,
                                           reg_model=self.reg_model,
                                           cea_obj=self.cea_obj,
                                           )
        
        self.upperStageEngine.estimate_all()
        self.firstStageEngine.estimate_all()

    def build_payload_bay(self):
        self.payloadBay = PayloadBayStructure(payloadRaidus=self.payloadBayParams["payloadRadius"],
                                              payloadHeight=self.payloadBayParams["payloadHeight"],
                                              payloadMass=self.payloadBayParams["payloadMass"],
                                              lowerSategeRadius=self.payloadBayParams["lowerStageRadius"],
                                              S_lower_rocket=0,
                                              verbose=False)        
        self.payloadBay.estimate_all()

    def build_upper_stage(self):
        coef_e2 = 0.06
        #old_coef_e2 = coef_e2
    
        for i in range(50):
            old_coef_e2 = coef_e2

            m_fairing = self.payloadBay.totalPayloadFairingMass
            m_pl = self.payloadBay.totalMass
            Isp_vac = self.upperStageEngine.IspVac
            deltaV = self.deltaV_upperStage
            #Isp_vac = 351.1

            propellantMass = self.calculate_propellant_mass_upper_stage(coef_e2, m_pl, m_fairing, deltaV, self.g0, Isp_vac)
            self.upperStageStructure = interStageStructure(oxName=self.upperStageStructureParams["oxName"],
                                                        fuelName=self.upperStageStructureParams["fuelName"],
                                                        propellantMass=propellantMass,
                                                        MR=self.upperStageEngine.MR,
                                                        radius=self.upperStageStructureParams["radius"],
                                                        tankPressure=self.upperStageStructureParams["tankPressure"],
                                                        maxEngineThrust=self.upperStageEngine.thrustVac,
                                                        lowerRadius=False,
                                                        upperMass=m_pl,
                                                        verbose=False
                                                            )
            self.upperStageStructure.estimate_all()

            dryMass = self.upperStageStructure.totalMass + self.upperStageEngine.totalMass * self.nEnginesUpperStage
            coef_e2 = dryMass/(dryMass + propellantMass)
            #print(self.upperStageStructure.totalMass)

            self.payloadBay.S_rocket = self.upperStageStructure.totalSurfaceArea
            self.build_payload_bay()

            diff_coef_e2 = np.abs(coef_e2 - old_coef_e2)
            if diff_coef_e2 <= 0.0001:
                break

        #print(f"Coeficiente Estrutural: {coef_e2}, Massa de Propelente: {propellantMass}")

        self.m_0_2 = propellantMass + m_pl + dryMass
        self.coef_e2 = coef_e2
        self.m_p_2 = propellantMass

        if diff_coef_e2 > 0.0001:
            self.m_p_2 = np.nan
            self.m_0_2 = np.nan
            self.glow = np.nan
            return
            #print("Não Convergiu")

    def calculate_propellant_mass_upper_stage(self, coef_e, m_pl, m_fairing, deltaV, g0, Isp_vac ):
        num = (m_pl - m_fairing) * (1 - math.exp( deltaV / (g0 * Isp_vac) ))
        den = (1 / (coef_e - 1)) * (1 - coef_e * (math.exp( deltaV / (g0 * Isp_vac) )))
        propellantMass = num/den
        return propellantMass

    def build_landing(self):
        Isp_sea = self.firstStageEngine.IspSea
        coef_e =  - math.exp(  (self.deltaV_landing / (Isp_sea * self.g0)))
        self.coef_e1_landing = coef_e
        #print(self.coef_e1_landing)
        return
    
    def build_first_stage(self):
        coef_e1 = 0.06
        Isp_vac = self.firstStageEngine.IspVac
        #Isp_vac = 278.4
        Isp_sea = self.firstStageEngine.IspSea
        coef_e1_landing =  np.exp(- (self.deltaV_landing / (Isp_sea * self.g0)))

        for i in range(50):
            # old_coef_e1 = coef_e1
            # m_s_1_num = 1 - (math.exp( (self.deltaV_firstStage)/ (Isp_sea *self.g0)))
            # m_s_1_den =   math.exp((self.deltaV_firstStage)/ (Isp_sea * self.g0)) - 1/coef_e1
            # m_s_1 = m_s_1_num / m_s_1_den * self.m_0_2

            # m_p_1 = m_s_1 * (1 - coef_e1) / coef_e1
            # #m_p_1_landing = m_s_1 * (1 - self.coef_e1_landing) / self.coef_e1_landing
            # #m_p_1_ascent = m_p_1 - m_p_1_landing

            old_coef_e1 = coef_e1
            m_s_1_num = 1 - (math.exp( (self.deltaV_firstStage)/ (Isp_sea *self.g0)))
            m_s_1_den =  (1/coef_e1_landing * math.exp((self.deltaV_firstStage)/ (Isp_sea * self.g0))) - 1/coef_e1
            m_s_1 = (m_s_1_num / m_s_1_den) * self.m_0_2

            m_p_1 = m_s_1 * (1 - coef_e1) / coef_e1
            m_p_1_landing = m_s_1 * (1 - coef_e1_landing) / coef_e1_landing
            m_p_1_ascent = m_p_1 - m_p_1_landing


            self.firstStageStructure = interStageStructure(oxName=self.firstStageStructureParams["oxName"],
                                                        fuelName=self.firstStageStructureParams["fuelName"],
                                                        propellantMass=m_p_1,
                                                        MR=self.upperStageEngine.MR,
                                                        radius=self.firstStageStructureParams["radius"],
                                                        tankPressure=self.firstStageStructureParams["tankPressure"],
                                                        maxEngineThrust=self.firstStageEngine.thrustVac * self.nEnginesFirstStage,
                                                        lowerRadius=False,
                                                        upperMass=m_p_1,
                                                        verbose=False
                                                            )
            self.firstStageStructure.estimate_all()

            dry_mass = self.firstStageStructure.totalMass + self.firstStageEngine.totalMass * self.nEnginesFirstStage
            coef_e1 = dry_mass / (dry_mass + m_p_1)

            diff_coef_1 = np.abs(coef_e1 - old_coef_e1)
            if diff_coef_1 <= 0.0001:
                break
        #print(f"Coeficiente Estrutural: {coef_e1}, Massa de Propelente: {m_p_1}")

        self.m_p_1 = m_p_1
        self.m_0_1 = dry_mass + m_p_1
        self.coef_e1 = coef_e1
        #print( m_p_1)
        self.glow = self.m_0_1 + self.m_0_2
        #print(self.glow)

        if diff_coef_1 > 0.0001:
            self.glow = np.nan
            #print("Não Convergiu")

    def build_all(self):
        self.build_engines()
        self.build_payload_bay()
        self.build_upper_stage()
        #self.build_landing()
        self.build_first_stage()
        self.calc_engine_dimensions_diff()
        self.totalHeight = self.upperStageStructure.totalHeight + self.firstStageStructure.totalHeight + self.payloadBay.totalHeight

    def calc_engine_dimensions_diff(self):
        """
        Calcula a diferenca entre o raio de saida do bocal e o raio do estágio
        para verificar se o motor é maior que o estágio
        """

        # Calculo de diferença para estágio superior
        raio_sup = self.upperStageStructure.radius
        raio_bocal_sup = self.upperStageEngine.nozzleDiam/2
        exp_ratio_sup = self.upperEngineParams["eps"]
        raio_saida_sup = raio_bocal_sup * math.sqrt(exp_ratio_sup)
        self.diff_raio_sup =  raio_sup - raio_saida_sup

        # Calculo de diferença para estágio inferior
        raio_inf = self.firstStageStructure.radius
        raio_bocal_inf = self.firstStageEngine.nozzleDiam/2
        exp_ratio_inf = self.firstEngineParams["eps"]
        raio_saida_inf = raio_bocal_inf * math.sqrt(exp_ratio_inf)
        self.diff_raio_inf = raio_inf - 3 * raio_saida_inf # Multipliquei por 3 aqui por conta de ter 9 motores e
        # pela geometria de organizaacao dos motores, 3 motores por raio

    def print_all_parameters(self):
        print( "*"*5 + " Payload Bay " + "*"*5) 
        self.payloadBay.print_all_parameters()
        print( "*"*5 + " Upper Stage Engine " + "*"*5) 
        self.upperStageEngine.print_all_parameters()
        print( "*"*5 + " Upper Stage Structure " + "*"*5) 
        self.upperStageStructure.print_all_parameters()
        print( "*"*5 + " First Stage Engine " + "*"*5) 
        self.firstStageEngine.print_all_parameters()
        print( "*"*5 + " First Stage Structure " + "*"*5) 
        self.firstStageStructure.print_all_parameters()
        print( "*"*5 + " Rocket Characteristics " + "*"*5) 
        print(f"Total Rocket Height: {self.totalHeight} [m]")
        print(f"Propellant Mass - Upper Stage: {self.m_p_2} [kg]")
        print(f"Propellant Mass - First Stage: {self.m_p_1} [kg]")
        print(f"Total Mass - Upper Stage: {self.m_0_2} [kg]")
        print(f"Total Mass - First Stage: {self.m_0_1} [kg]")
        print(f"GLOW: {self.glow} [kg]")




if __name__ == "__main__":
    import joblib
    import time
    from rocketcea.cea_obj_w_units import CEA_Obj

    #reg_path = '/Users/pdcos/Documents/Estudos/Mestrado/Tese/Implementação da Tese do Jentzsch/rocket_optimization_implementation/model/engines/decision_tree_model.pkl'
    #reg_path = '/home/ubuntu/Mestrado/modelo_foguete/model/engines/decision_tree_model.pkl'
    reg_path = '/home/ubuntu/Mestrado/modelo_foguete_corrigido/improve_exec_speed/data/DecisionTreeRegressor_score_1.0.joblib'
    reg_model = joblib.load(reg_path)
    #reg_model = False
    cea_obj = ceaObj = CEA_Obj( oxName='LOX', fuelName='RP-1', pressure_units='Pa', cstar_units='m/s', temperature_units='K')


    # engineParams = {"oxName": "LOX",
    #                 "fuelName": "RP-1",
    #                 "combPressure": 11.5 * 1e6,
    #                 "MR": 2.8,
    #                 "nozzleDiam": 0.23125,
    #                 "eps": 180}

    # engineParamsFirst = {"oxName": "LOX",
    #                 "fuelName": "RP-1",
    #                 "combPressure": 11.5 * 1e6,
    #                 "MR": 2.8,
    #                 "nozzleDiam": 0.23125,
    #                 "eps": 25}

    # payloadBayParams = {"payloadHeight": 6.7,
    #                     "payloadRadius": 4.6/2,
    #                     "payloadMass": 7500,
    #                     "lowerStageRadius": 2.1,
    #                     "lowerRocketSurfaceArea": 0} # 0 porque ainda nao temos esse valor

    # upperStageStructureParams = {"oxName": "LOX",
    #                              "fuelName": "RP1",
    #                              "MR": 2.8,
    #                              "tankPressure": 0.1,
    #                              "radius": 2.1,
    #                             } # 0 porque ainda nao temos esse valor
    # lowerStageStructureParams = {"oxName": "LOX",
    #                             "fuelName": "RP1",
    #                             "MR": 2.8,
    #                             "tankPressure": 0.1,
    #                             "radius": 2.8,
    #                         } # 0 porque ainda nao temos esse valor


    # engineParams = {"oxName": "LOX",
    #                 "fuelName": "RP-1",
    #                 "combPressure": 3.96769844e6,
    #                 "MR": 4.31244790,
    #                 "nozzleDiam": 4.31920087e-1,
    #                 "eps": 8.40074954e1}

    # engineParamsFirst = {"oxName": "LOX",
    #                 "fuelName": "RP-1",
    #                 "combPressure": 2.44865031e6,
    #                 "MR": 5.28717125,
    #                 "nozzleDiam": 4.15087055e-1,
    #                 "eps": 1.03948044e2}

    # payloadBayParams = {"payloadHeight": 6.7,
    #                     "payloadRadius": 4.6/2,
    #                     "payloadMass": 4850,
    #                     "lowerStageRadius": 9.50135280,
    #                     "lowerRocketSurfaceArea": 0} # 0 porque ainda nao temos esse valor

    # upperStageStructureParams = {"oxName": "LOX",
    #                              "fuelName": "RP1",
    #                              "MR": 4.31244790,
    #                              "tankPressure": 0.1,
    #                              "radius": 9.50135280,
    #                             } # 0 porque ainda nao temos esse valor
    # lowerStageStructureParams = {"oxName": "LOX",
    #                             "fuelName": "RP1",
    #                             "MR": 5.28717125,
    #                             "tankPressure": 0.1,
    #                             "radius": 6.27899536,
    #                         } # 0 porque ainda nao temos esse valor

    # time_start = time.time()
    # for i in range(1000):
    #     rocket_model = RocketModel(upperEngineParams=engineParams,
    #                             firstEngineParams=engineParamsFirst,
    #                             payloadBayParams=payloadBayParams,
    #                             upperStageStructureParams=upperStageStructureParams,
    #                             firstStageStructureParams = lowerStageStructureParams,
    #                             deltaV_upperStage=9000,
    #                             deltaV_landing=2000,
    #                             deltaV_firstStage=3000,
    #                             nEnginesUpperStage=1,
    #                             nEnignesFirstStage=9,
    #                             reg_model=reg_model,
    #                             cea_obj=cea_obj,
    #                             )

    #     rocket_model.build_all()
    #     #rocket_model.print_all_parameters()
    # time_end = time.time()
    # print(f'Tempo: {time_end - time_start}')
    # #rocket_model.print_all_parameters()
    # print(rocket_model.glow)


    engineParams = {"oxName": "LOX",
                    "fuelName": "RP-1",
                    "combPressure": 9.72 * 1e6,
                    "MR": 2.36,
                    "nozzleDiam": 0.23125,
                    "eps": 117}

    engineParamsFirst = {"oxName": "LOX",
                    "fuelName": "RP-1",
                    "combPressure": 9.72 * 1e6,
                    "MR": 2.34,
                    "nozzleDiam": 0.23125,
                    "eps": 21.4}

    payloadBayParams = {"payloadHeight": 6.7,
                        "payloadRadius": 4.6/2,
                        "payloadMass": 4850,
                        "lowerStageRadius": 2.1,
                        "lowerRocketSurfaceArea": 0} # 0 porque ainda nao temos esse valor

    upperStageStructureParams = {"oxName": "LOX",
                                 "fuelName": "RP1",
                                 "MR": 2.36,
                                 "tankPressure": 0.1,
                                 "radius": 1.83,
                                } # 0 porque ainda nao temos esse valor
    lowerStageStructureParams = {"oxName": "LOX",
                                "fuelName": "RP1",
                                "MR": 2.34,
                                "tankPressure": 0.1,
                                "radius": 1.83,
                            } # 0 porque ainda nao temos esse valor


    rocket_model = RocketModel(upperEngineParams=engineParams,
                            firstEngineParams=engineParamsFirst,
                            payloadBayParams=payloadBayParams,
                            upperStageStructureParams=upperStageStructureParams,
                            firstStageStructureParams = lowerStageStructureParams,
                            deltaV_upperStage=9000,
                            deltaV_landing=2000,
                            deltaV_firstStage=3000,
                            nEnginesUpperStage=1,
                            nEnignesFirstStage=9,
                            reg_model=reg_model,
                            cea_obj=cea_obj,
                            )

    rocket_model.build_all()
    rocket_model.print_all_parameters()

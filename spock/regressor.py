import numpy as np
from .simsetup import init_sim_parameters
from .feature_functions import get_extended_tseries

class StabilityRegressor():
    def __init__(self):
        pass

    def check_errors(self, sim):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets") 
        
    def predict_stable(self, sim):
        t_inst = self.predict_instability_time(sim)
        minP = np.min([p.P for p in sim.particles[1:sim.N_real]])
        if t_inst >= 1e9*minP:
            return 1
        else:
            return 0
        
    def predict_from_tseries(self, triotseries):
        # predict an instability time for each trio in triotseries
        trio_tinsts = np.zeros(len(triotseries))
        for i, tseries_array in enumerate(triotseries):
            trio_tinsts[i] = self.evaluate_model(tseries_array)
        return trio_tinsts

    def evaluate_model(self, tseries_array):
        # plug in function
        return 0 

    def predict_instability_time(self, sim):
        sim = sim.copy()
        init_sim_parameters(sim)
        self.check_errors(sim)
        
        trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios   
        args = [10000, 80, trios]
        triotseries, stable = get_extended_tseries(sim, args)
        
        if stable == False:
            return sim.t
       
        trio_tinsts = self.predict_from_tseries(triotseries)
        return trio_tinsts.min()


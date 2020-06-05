import numpy as np
from .simsetup import init_sim_parameters

class StabilityRegressor():
    def __init__(self):
        pass

    def check_errors(self, sim):
        if sim.N_real < 4:
            raise AttributeError("SPOCK Error: SPOCK only applicable to systems with 3 or more planets") 
        
    def predict_stable(self, sim):
        sim = sim.copy()
        init_sim_parameters(sim)
        self.check_errors(sim)
        
        trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios   
        featureargs = [10000, 80, trios]
        triotseries, stable = get_tseries(sim, args)
        
        if stable == False:
            return 0
       
        trioprobs = self.predict_from_tseries(triotseries)
        return trioprobs.min()          # minimum prob among all trios tested

    def predict_from_tseries(self, triotseries):
        trioprobs = np.zeros(len(triotseries))
        for i, tseries_array in enumerate(triotseries):
            trioprobs[i] = self.evaluate_model(tseries_array)
        return trioprobs

    def evaluate_model(self, tseries_array):
        # plug in function
        return 0 

    def predict_instability_time(self, sim, tmax=None):
        sim = sim.copy()
        init_sim_parameters(sim)
        self.check_errors(sim)
        
        trios = [[i,i+1,i+2] for i in range(1,sim.N_real-2)] # list of adjacent trios   
        featureargs = [10000, 80, trios]
        triotseries, stable = get_extended_tseries(sim, args)
        # should return tinst if unstabble so we can  return that 
        if stable == False:
            return 0
       
        trioprobs = self.predict_from_tseries(triotseries)
            stable = False
            return sim.t, stable
            
        stable = True
        return tmax, stable


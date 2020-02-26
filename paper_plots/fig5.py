#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append('../')


import numpy as np
import rebound
import matplotlib.pyplot as plt
import matplotlib
import random
import dill
import sys
import pandas as pd
import spock
from spock import StabilityRegression

try:
    plt.style.use('paper')
except:
    pass

start = int(sys.argv[1])*437
end = start + 437
if end > 17500:
    end = 17500

spockoutfile = 'alysa-datafiles/spockprobstesttrio{}_to_{}_v3.npz'.format(start, end)

stride = 10
nsim_list = np.arange(0, 17500)

nsim_list = nsim_list[start:end:stride]


#######################################################################
## read initial condition file

infile_delta_2_to_10 = 'alysa-datafiles/initial_conditions_delta_2_to_10.npz'
infile_delta_10_to_13 = 'alysa-datafiles/initial_conditions_delta_10_to_13.npz'

ic1 = np.load(infile_delta_2_to_10)
ic2 = np.load(infile_delta_10_to_13)

m_star = ic1['m_star'] # mass of star
m_planet = ic1['m_planet'] # mass of planets
rh = (m_planet/3.) ** (1./3.)

Nbody = ic1['Nbody'] # number of planets
year = 2.*np.pi # One year in units where G=1
tf = ic1['tf'] # end time in years

a_init = np.concatenate([ic1['a'], ic2['a']], axis=1) # array containing initial semimajor axis for each delta,planet
f_init = np.concatenate([ic1['f'], ic2['f']], axis=1) # array containing intial longitudinal position for each delta, planet, run

#######################################################################
## create rebound simulation and predict stability for each system in nsim_list


infile_delta_2_to_10 = 'alysa-datafiles/initial_conditions_delta_2_to_10.npz'
infile_delta_10_to_13 = 'alysa-datafiles/initial_conditions_delta_10_to_13.npz'

outfile_nbody_delta_2_to_10 = 'alysa-datafiles/merged_output_files_delta_2_to_10.npz'
outfile_nbody_delta_10_to_13 = 'alysa-datafiles/merged_output_files_delta_10_to_13.npz'

## load hill spacing

ic_delta_2_to_10 = np.load(infile_delta_2_to_10)
ic_delta_10_to_13 = np.load(infile_delta_10_to_13)

delta_2_to_10 = ic_delta_2_to_10['delta']
delta_10_to_13 = ic_delta_10_to_13['delta']

delta = np.hstack((delta_2_to_10, delta_10_to_13))
delta=delta[start:end:stride]

## load rebound simulation first close encounter times

nbody_delta_2_to_10 = np.load(outfile_nbody_delta_2_to_10)
nbody_delta_10_to_13 = np.load(outfile_nbody_delta_10_to_13)

t_exit_delta_2_to_10 = nbody_delta_2_to_10['t_exit']/(0.99)**(3./2)
t_exit_delta_10_to_13 = nbody_delta_10_to_13['t_exit']/(0.99)**(3./2)

t_exit = np.hstack((t_exit_delta_2_to_10, t_exit_delta_10_to_13))
t_exit = t_exit[start:end:stride]

df = pd.DataFrame(np.array([nsim_list, delta, t_exit]).T, columns=['nsim', 'delta', 't_exit'])
df.head()

model = StabilityRegression()

def pred(nsim):
    sim = rebound.Simulation()
    sim.add(m=m_star)
    sim.G = 4*np.pi**2
    for i in range(Nbody): # add the planets
        sim.add(m=m_planet, a=a_init[i, nsim], f=f_init[i, nsim])
        print(a_init[i, nsim])
    sim.move_to_com()
    sim.init_megno(seed=0)
    sim.integrator="whfast"
    sim.dt = 0.07*sim.particles[1].P
    prob = model.predict(sim, samples=10000)
    avgprob = 10**np.average(np.log10(prob))
    print("Done", nsim, "with", avgprob, flush=True)
    return prob


#init_process()


#pred(-1)


# %%time
#res = pool.map(pred, nsim_list)
res = list(map(pred, nsim_list))


np.savez(spockoutfile, nsim_list=nsim_list, probs=np.array(res))

exit()


infile_delta_2_to_10 = 'alysa-datafiles/initial_conditions_delta_2_to_10.npz'
infile_delta_10_to_13 = 'alysa-datafiles/initial_conditions_delta_10_to_13.npz'

outfile_nbody_delta_2_to_10 = 'alysa-datafiles/merged_output_files_delta_2_to_10.npz'
outfile_nbody_delta_10_to_13 = 'alysa-datafiles/merged_output_files_delta_10_to_13.npz'

spockoutfile_ml_probs = 'alysa-datafiles/stability-probs-sims-spockalltrios.npz'

## load hill spacing

ic_delta_2_to_10 = np.load(infile_delta_2_to_10)
ic_delta_10_to_13 = np.load(infile_delta_10_to_13)

delta_2_to_10 = ic_delta_2_to_10['delta']
delta_10_to_13 = ic_delta_10_to_13['delta']

delta = np.hstack((delta_2_to_10, delta_10_to_13))
delta=delta[start:end:stride]

## load rebound simulation first close encounter times

nbody_delta_2_to_10 = np.load(outfile_nbody_delta_2_to_10)
nbody_delta_10_to_13 = np.load(outfile_nbody_delta_10_to_13)

t_exit_delta_2_to_10 = nbody_delta_2_to_10['t_exit']/(0.99)**(3./2)
t_exit_delta_10_to_13 = nbody_delta_10_to_13['t_exit']/(0.99)**(3./2)

t_exit = np.hstack((t_exit_delta_2_to_10, t_exit_delta_10_to_13))
t_exit = t_exit[start:end:stride]

log_t_exit = np.log10(t_exit)
survival_1e9 = log_t_exit >= 9

survival_true = survival_1e9 == True
survival_false = survival_1e9 == False

## load machine learning stability probabilities

spockml_data = np.load(spockoutfile)
spockprobs = spockml_data['probs']

delta.shape


# #  TPR


thresh = 0.79
(spockprobs[survival_true] > thresh).sum()/survival_true.sum()


# # FPR


(spockprobs[survival_false] > thresh).sum()/survival_false.sum()


survival_false.sum()


from matplotlib.lines import Line2D

fig, axarr = plt.subplots(2, 1, figsize=(16,8), sharex=True)
plt.subplots_adjust(hspace=0, wspace=0)

for ax in axarr.flatten():
    ax.set_rasterization_zorder(1)

ms=5

stride = 4
axarr[0].plot([np.min(delta), np.max(delta)], [9,9], '--k')
axarr[0].plot(delta[survival_true][::stride],log_t_exit[survival_true][::stride],'ob', alpha=0.2, markersize=ms,rasterized=True,label='Stable', zorder=0)
axarr[0].plot(delta[survival_false][::stride],log_t_exit[survival_false][::stride],'or', alpha=0.2, markersize=ms,rasterized=True,label='Unstable', zorder=0)

axarr[1].plot(delta[survival_true][::stride], spockprobs[survival_true][::stride],'ob', alpha=0.2, markersize=ms*2,rasterized=True, zorder=0)
axarr[1].plot(delta[survival_false][::stride], spockprobs[survival_false][::stride],'or', alpha=0.2, markersize=ms, rasterized=True, zorder=0)

axarr[1].set_xlabel("Interplanetary Separation (Hill radii)", fontsize=18)
axarr[1].set_xlim([1.9, 13.1])

axarr[1].set_ylabel("SPOCK Stability \nProbability", fontsize=18)
axarr[0].set_ylabel("Log Instability Time\n (Orbits)", fontsize=18)

axarr[0].legend(loc=2, markerscale=3)

legend_elements = [Line2D([0], [0], marker='o', lw=0, c='b', markerfacecolor='b', label='Stable', markersize=10),
                   Line2D([0], [0], marker='o', lw=0, c='r', markerfacecolor='r', label='Unstable', markersize=10)]
axarr[0].legend(handles=legend_elements, loc='upper left', fontsize=24)


plt.savefig('alysa.pdf',bbox_inches='tight')



from NeuronModelClass import NeuronModel
from NrnHelper import *
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np
from currentscape.currentscape import plot_currentscape
import pandas as pd
import os
import datetime
import csv


class Na12HH:
    def __init__(self,na12name = 'na12WT',mut_name= 'na12WT',  na12mechs = ['na12','na12mut'],na16name = 'na16WT',na16mut_name ='na16WT', na16mechs = ['na16','na16'], params_folder = './params/na12HMM_HOF_params/', 
                 nav12=1,nav16=1,K=1,KT=1,KP=1,somaK=1,ais_ca = 1,ais_Kca = 1,soma_na16=1,soma_na12 = 1,node_na = 1,plots_folder = './Plots/',pfx='testprefix', ais_nav16_fac=1,ais_nav12_fac=1, dend_nav12=1,
                 update = None, fac=None): 
        K = 1 
        node_na = 0.5 
        self.l5mdl = NeuronModel(nav12=nav12, nav16=nav16,axon_K = K,axon_Kp = KP,axon_Kt = KT,soma_K = somaK,
                                 ais_ca = ais_ca,ais_KCa=ais_Kca,soma_nav16=soma_na16,soma_nav12 = soma_na12,node_na = node_na, 
                                 ais_nav16_fac=ais_nav16_fac,ais_nav12_fac=ais_nav12_fac, 
                                 dend_nav12=dend_nav12,
                                 update = update, 
                                 na12name = na12name,
                                 na12mut_name = mut_name,
                                 na12mechs = na12mechs,
                                 na16name = na16name,
                                 na16mut_name = na16mut_name,
                                 na16mechs = na16mechs,
                                 params_folder=params_folder,
                                 fac=fac
                                 ) 
        
        self.plot_folder = plots_folder 
        self.plot_folder = f'{plots_folder}'
        Path(self.plot_folder).mkdir(parents=True, exist_ok=True)
        self.pfx = pfx


    def update_gfactor(self,gbar_factor = 1):
        update_mod_param(self.l5mdl, self.mut_mech, gbar_factor, gbar_name='gbar')

    def plot_stim(self,stim_amp = 0.5,dt = 0.02,clr = 'black',plot_fn = 'step',axs = None,rec_extra = False, stim_dur = 500):
        self.dt = dt
        if not axs:
            fig,axs = plt.subplots(1,figsize=(cm_to_in(8),cm_to_in(7.8)))
        self.l5mdl.init_stim(stim_dur = stim_dur, amp=stim_amp )
        if rec_extra:
            Vm, I, t, stim,extra_vms = self.l5mdl.run_model(dt=dt,rec_extra = rec_extra)
            self.extra_vms = extra_vms
        else:
            Vm, I, t, stim = self.l5mdl.run_model(dt=dt)
            
        self.volt_soma = Vm
        self.I = I
        self.t = t
        self.stim = stim

        ap_t = (t/dt)*1000 ##Get timesteps
        
        axs.plot(t,Vm, label='Vm', color=clr,linewidth=0.5) ##TF031424 changed linewidth
        axs.locator_params(axis='x', nbins=5)
        axs.locator_params(axis='y', nbins=8)
        file_path_to_save=f'{self.plot_folder}{plot_fn}.pdf'
        return ap_t, Vm
    
    #Function for getting raw data from WT to superimpose under mut plots
    def get_stim_raw_data(self,stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=1600,sim_config = {
            }):
        
        ap_initiation = []
        ap_threshold =10

        self.dt = dt
        self.l5mdl.init_stim(stim_dur = stim_dur, amp=stim_amp)
        if rec_extra:
            Vm, I, t, stim,extra_vms = self.l5mdl.run_sim_model(dt=dt,rec_extra = rec_extra, sim_config=sim_config)
            self.extra_vms = extra_vms
        else:
            Vm, I, t, stim, ionic = self.l5mdl.run_sim_model(dt=dt, sim_config=sim_config)
            
        return Vm, I, t, stim

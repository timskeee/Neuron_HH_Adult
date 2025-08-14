# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:07:44 2021

@author: bensr
"""
import argparse
import numpy as np
from vm_plotter import *
from neuron import h
import os
import sys
from NrnHelper import *


class NeuronModel:
    def __init__(self,ais_nav16_fac, ais_nav12_fac, mod_dir ='./Neuron_Model_12HH16HH/',update = None,na12name = 'na12WT',
                 na12mut_name = 'na12WT',na12mechs = ['na12','na12mut'],na16name = 'na16HH',na16mut_name = 'na16HH',
                 na16mechs=['na16','na16mut'],params_folder = './params/',nav12=1,nav16=1,dend_nav12=1,soma_nav12=1,
                 dend_nav16=1,soma_nav16=1,ais_nav12=1,ais_nav16=1,ais_ca = 1,ais_KCa = 1,axon_Kp=1,axon_Kt =1,axon_K=1,
                 axon_Kca =1,axon_HVA = 1,axon_LVA = 1,node_na = 1,soma_K=1,dend_K=1,gpas_all=1,fac=None):
        run_dir = os.getcwd()

        os.chdir(mod_dir)
        self.h = h  # NEURON h
        print(f'running model at {os.getcwd()} run dir is {run_dir}')
        print (f'There is {nav16} of WT nav16')
        print(f'There is {nav12} of WT nav12')
        h.load_file("runModel.hoc")
        self.soma_ref = h.root.sec
        self.soma = h.secname(sec=self.soma_ref)
        self.sl = h.SectionList()
        self.sl.wholetree(sec=self.soma_ref)
        self.nexus = h.cell.apic[66]
        self.dist_dend = h.cell.apic[91]
        self.ais = h.cell.axon[0]
        self.axon_proper = h.cell.axon[1]
        
        h.dend_na12 = 2.48E-03 * dend_nav12
        h.dend_na16 = 0 
        dend_nav16=5.05E-06 * dend_nav16
        h.dend_k = 0.0043685576 * dend_K
        h.soma_na12 = 3.24E-02 * soma_nav12 
        h.soma_na16 = 7.88E-02 * soma_nav16
        h.soma_K = 0.21330453 * soma_K
        h.ais_na16 = ais_nav16_fac * ais_nav16
        h.ais_na12 = ais_nav12_fac * ais_nav12 
        h.ais_ca = 0.0010125926 * ais_ca
        h.ais_KCa = 0.0009423347 * ais_KCa
        h.node_na = 0.9934221 * node_na
        h.axon_KP = 0.43260124 * axon_Kp
        h.axon_KT = 1.38801 * axon_Kt
        h.axon_K = 0.89699364 *2.1* axon_K
        h.axon_LVA = 0.00034828275 * axon_LVA
        h.axon_HVA = 1.05E-05 * axon_HVA
        h.axon_KCA = 0.4008224 * axon_Kca
        h.gpas_all = 1.34E-05 * gpas_all
        h.cm_all = 1.6171424
        h.dend_na12 = h.dend_na12 * nav12 * dend_nav12
        h.soma_na12 = h.soma_na12 * nav12 * soma_nav12
        if nav12 !=0:
            h.ais_na12 = (h.ais_na12 * ais_nav12)/nav12 
        else:
            h.ais_na12 = h.ais_na12 *ais_nav12
        
        if nav16 !=0:
            h.ais_na16 = (h.ais_na16 * ais_nav16)/nav16
        else:
            h.ais_na12 = h.ais_na16 * ais_nav16

        h.dend_na16 = h.dend_na16 * nav16 * dend_nav16
        h.soma_na16 = h.soma_na16 * nav16 * soma_nav16
        h.working()
        os.chdir(run_dir)

        if update:
            update_param_value(self,['SKv3_1'],'mtaumul',6) 
            multiply_param(self,['SKv3_1'],'mtaumul',0.85)
            self.na12wt_mech = [na12mechs[0]] 
            self.na12mut_mech = [na12mechs[1]]

            self.na16wt_mech = [na16mechs[0]] 
            self.na16mut_mech = [na16mechs[1]] 
            self.na16mechs = na16mechs

            self.h.working()                                                 
            p_fn_na12 = f'{params_folder}{na12name}.txt'  
            p_fn_na12_mech = f'{params_folder}{na12mut_name}.txt'
            print(f'using wt_file params {na12name}')
            self.na12_p = update_mech_from_dict(self, p_fn_na12, self.na12wt_mech) 
            print(f'using mut_file params {na12mut_name}')
            self.na12_pmech = update_mech_from_dict(self, p_fn_na12_mech, self.na12mut_mech) 
            
            update_mod_param(self,['na12','na12mut'],nav12)
            
            p_fn_na16 = f'{params_folder}{na16name}.txt'
            p_fn_na16_mech = f'{params_folder}{na16mut_name}.txt'
            
            print(f'using na16wt_file params {na16name}')
            self.na16_p = update_mech_from_dict(self, p_fn_na16,self.na16wt_mech) ###
           
            
            print(f'using na16mut_file params {na16mut_name}')
            self.na16_pmech = update_mech_from_dict(self, p_fn_na16_mech,self.na16mut_mech) ###
            
            # add nav16 only to first 20 microns of dendrites, otherwise gbar 0
            update_mod_param(self,['na16','na16mut'],nav16)
            for sec in self.h.allsec():
                if 'dend' in sec.name() or 'apic' in sec.name():
                    for seg in sec:
                        if self.h.distance(sec(0.5), seg.x) <= 20:
                            for mech in ['na16', 'na16mut']:
                                if hasattr(seg, mech):
                                    setattr(getattr(seg, mech), 'gbar', dend_nav16)#dend_nav16)
                        else:
                            for mech in ['na16', 'na16mut']:
                                if hasattr(seg, mech):
                                    setattr(getattr(seg, mech), 'gbar', 0)            
            
        
    
    def init_stim(self, sweep_len = 2000, stim_start = 200, stim_dur = 1700, amp = 0.5, dt = 0.1): ##TF021425 long sweep to get smoother FIs
        h("st.del = " + str(stim_start))
        h("st.dur = " + str(stim_dur))
        h("st.amp = " + str(amp))
        h.tstop = sweep_len
        h.dt = dt

    def init_stim_dend(self, sweep_len = 150, stim_start = 30, stim_dur = 100, amp = 0.5, dt = 0.1):
        h("st_dend.del = " + str(stim_start))
        h("st_dend.dur = " + str(stim_dur))
        h("st_dend.amp = " + str(amp))
        h.tstop = sweep_len
        h.dt = dt
    
    def start_stim(self,tstop = 800, start_Vm = -72):
        h.finitialize(start_Vm)
        h.tstop = tstop
        
    def run_model2(self, stim_start = 100, stim_dur = 0.2, amp = 0.3, dt= 0.1,rec_extra = False): # works in combinition with stim_start for working with physiological stimultion
        h.dt=dt
        h("st.del = " + str(stim_start))
        h("st.dur = " + str(stim_dur))
        h("st.amp = " + str(amp))
        timesteps = int(stim_dur/h.dt) # changed from h.tstop to stim_dur
        Vm = np.zeros(timesteps)
        I = {}
        I['Na'] = np.zeros(timesteps)
        I['Ca'] = np.zeros(timesteps)
        I['K'] = np.zeros(timesteps)
        stim = np.zeros(timesteps)
        t = np.zeros(timesteps)
        if rec_extra:
            
            extra_Vms = {}
            extra_Vms['ais'] = np.zeros(timesteps)
            extra_Vms['nexus'] = np.zeros(timesteps)
            extra_Vms['dist_dend'] = np.zeros(timesteps)
            extra_Vms['axon'] = np.zeros(timesteps)

        for i in range(timesteps):
            Vm[i] = h.cell.soma[0].v
            I['Na'][i] = h.cell.soma[0](0.5).ina
            I['Ca'][i] = h.cell.soma[0](0.5).ica
            I['K'][i] = h.cell.soma[0](0.5).ik
            stim[i] = h.st.amp
            t[i] = (stim_start + i*h.dt) / 1000 #after each run_modl2 call, the stim_start is updated to the current time
            if rec_extra:
                nseg = int(self.h.L/10)*2 +1  # create 19 segments from this axon section
                ais_end = 10/nseg # specify the end of the AIS as halfway down this section
                ais_mid = 4/nseg # specify the middle of the AIS as 1/5 of this section 
                extra_Vms['ais'][i] = self.ais(ais_mid).v
                extra_Vms['nexus'][i] = self.nexus(0.5).v
                extra_Vms['dist_dend'][i] = self.dist_dend(0.5).v
                extra_Vms['axon'][i]=self.axon_proper(0.5).v
            h.fadvance()
        if rec_extra:
            return Vm, I, t, stim,extra_Vms
        else:
            return Vm, I, t, stim
        
    def run_model(self, start_Vm = -72, dt= 0.1,rec_extra = False):
        h.dt=dt
        h.finitialize(start_Vm)
        timesteps = int(h.tstop/h.dt) # change later to h.tstop

        Vm = np.zeros(timesteps)
        I = {}
        I['Na'] = np.zeros(timesteps)
        I['Ca'] = np.zeros(timesteps)
        I['K'] = np.zeros(timesteps)
        stim = np.zeros(timesteps)
        t = np.zeros(timesteps)
        if rec_extra:
            
            extra_Vms = {}
            extra_Vms['ais'] = np.zeros(timesteps)
            extra_Vms['nexus'] = np.zeros(timesteps)
            extra_Vms['dist_dend'] = np.zeros(timesteps)
            extra_Vms['axon'] = np.zeros(timesteps)

        for i in range(timesteps):
            Vm[i] = h.cell.soma[0].v
            I['Na'][i] = h.cell.soma[0](0.5).ina
            I['Ca'][i] = h.cell.soma[0](0.5).ica
            I['K'][i] = h.cell.soma[0](0.5).ik
            stim[i] = h.st.amp
            t[i] = i*h.dt / 1000
            if rec_extra:
                nseg = int(self.h.L/10)*2 +1  # create 19 segments from this axon section
                ais_end = 10/nseg # specify the end of the AIS as halfway down this section
                ais_mid = 4/nseg # specify the middle of the AIS as 1/5 of this section 
                extra_Vms['ais'][i] = self.ais(ais_mid).v
                extra_Vms['nexus'][i] = self.nexus(0.5).v
                extra_Vms['dist_dend'][i] = self.dist_dend(0.5).v
                extra_Vms['axon'][i]=self.axon_proper(0.5).v
            h.fadvance()
        if rec_extra:
            return Vm, I, t, stim,extra_Vms
        else:
            return Vm, I, t, stim
        
    def run_sim_model(self, start_Vm = -72, dt= 0.1, sim_config = {
        #changing to get different firing at different points along neuron TF 011624
                'section' : 'soma',
                'segment' : 0.5,
                'section_num' : 0,                
                'currents'  :['ina','ica','ik'],
                'ionic_concentrations' :["cai", "ki", "nai"]
            }):        
        h.dt=dt
        h.finitialize(start_Vm)
        timesteps = int(h.tstop/h.dt)
        current_types = sim_config['currents']
        ionic_types = sim_config['ionic_concentrations']
        Vm = np.zeros(timesteps, dtype=np.float64)
        I = {current_type: np.zeros(timesteps, dtype=np.float64) for current_type in current_types}
        ionic = {ionic_type : np.zeros(timesteps,dtype=np.float64) for ionic_type in ionic_types}

        stim = np.zeros(timesteps, dtype=np.float64)
        t = np.zeros(timesteps, dtype=np.float64)
        section = sim_config['section']
        section_number = sim_config['section_num']
        segment = sim_config['segment']
        volt_var  = "h.cell.{section}[{section_number}]({segment}).v".format(section=section, section_number=section_number,segment=segment)
        curr_vars={}
        curr_vars = {current_type : "h.cell.{section}[{section_number}]({segment}).{current_type}".format(section=section, section_number=section_number, segment=segment, current_type=current_type) for current_type in current_types}
        ionic_vars = {ionic_type : "h.cell.{section}[{section_number}]({segment}).{ionic_type}".format(section=section , section_number=section_number, segment=segment, ionic_type=ionic_type) for ionic_type in ionic_types}
        for i in range(timesteps):
           
            Vm[i] =eval(volt_var)
             
            try :
                for current_type in current_types:
                    I[current_type][i] = eval(curr_vars[current_type])

                #getting the ionic concentrations
                for ionic_type in ionic_types:
                    ionic[ionic_type][i] = eval(ionic_vars[ionic_type])
            except Exception as e:
                print(e)
                print("Check the config files for the correct Attribute")
                sys.exit(1)

            stim[i] = h.st.amp
            t[i] = i*h.dt / 1000

            h.fadvance()

        return Vm, I, t, stim, ionic
    
    def plot_crazy_stim(self, stim_csv, stim_duration=None):
        if not stim_duration:
            stim_duration = 0.2 #ms
      


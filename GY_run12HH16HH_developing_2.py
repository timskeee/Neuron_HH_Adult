from NeuronModelClass import NeuronModel
from NrnHelper import *
import NrnHelper as nh
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np
from Na12HMMModel_TF import *
import Na12HMMModel_TF as tf
import os
import efel_feature_extractor as ef
from currentscape.currentscape import plot_currentscape
import logging
import pandas as pd
# import Document as doc
# import Tim_ng_functions as nf

sim_config_soma = {
                'section' : 'soma',
                'segment' : 0.5,
                'section_num': 0,
                'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ihcn_Ih','ik_SK_E2','ik_SKv3_1'],
                'current_names' : ['Ih','SKv3_1','Na16 WT','Na16 WT','Na12','Na12 MUT','pas'],
                'ionic_concentrations' :["cai", "ki", "nai"]
                }
sim_config_ais = {
                'section' : 'axon',
                'segment' : 0.1,
                'section_num': 0,
                #'currents' : ['ina','ica','ik'],
                #'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ihcn_Ih','ik_SK_E2','ik_SKv3_1'], #Somatic
                'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ik_SK_E2','ik_SKv3_1'], #AIS (no Ih)
                #'currents'  : ['ica_Ca_HVA','ica_Ca_LVAst','ik_SKv3_1','ik_SK_E2','na16.ina_ina','na16mut.ina_ina','na12.ina_ina','na12mut.ina_ina','i_pas'],
                #'currents'  : ['ihcn_Ih','ik_SKv3_1','na16.ina_ina','na16mut.ina_ina','na12.ina_ina','na12mut.ina_ina','i_pas'],
                'current_names' : ['Ih','SKv3_1','Na16 WT','Na16 WT','Na12','Na12 MUT','pas'],
                #'ionic_concentrations' :["cai", "ki", "nai"]
                'ionic_concentrations' :["ki", "nai"]
                }
# 3
sim_config_basaldend = {
                'section' : 'dend',
                'segment' : 0.5,
                'section_num': 70,
                #'currents' : ['ina','ica','ik'],
                #'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ihcn_Ih','ik_SK_E2','ik_SKv3_1'], #Somatic
                #'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ik_SK_E2','ik_SKv3_1'], #AIS (no Ih)
                #'currents'  : ['ica_Ca_HVA','ica_Ca_LVAst','ik_SKv3_1','ik_SK_E2','na16.ina_ina','na16mut.ina_ina','na12.ina_ina','na12mut.ina_ina','i_pas'],
                'currents'  : [], #dend (no Ih, no ik_SKv3_1)
                'current_names' : ['Ih','Na16 WT','Na16 WT','Na12','Na12 MUT','pas'],
                #'ionic_concentrations' :["cai", "ki", "nai"]
                'ionic_concentrations' :[]
                }
#4
sim_config_nexus = {
                'section' : 'apic',
                'segment' : 0,
                'section_num': 77,
                #'currents' : ['ina','ica','ik'],
                #'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ihcn_Ih','ik_SK_E2','ik_SKv3_1'], #Somatic
                'currents'  : ['ik_SKv3_1'], #Nexus
                #'currents'  : ['ica_Ca_HVA','ica_Ca_LVAst','ik_SKv3_1','ik_SK_E2','na16.ina_ina','na16mut.ina_ina','na12.ina_ina','na12mut.ina_ina','i_pas'],
                
                'current_names' : ['Ih','SKv3_1','Na16 WT','Na16 WT','Na12','Na12 MUT','pas'],
                #'ionic_concentrations' :["cai", "ki", "nai"]
                'ionic_concentrations' :["ki", "nai"]
                }
#5
sim_config_apicaldend = {
                'section' : 'apic',
                'segment' : 0.5,
                'section_num': 90,
                # 'currents' : ['ina','ica','ik'],
                #'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ihcn_Ih','ik_SK_E2','ik_SKv3_1'], #Somatic
                'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ik_SKv3_1','ihcn_Ih'], #AIS (no Ih)
                #'currents'  : ['ica_Ca_HVA','ica_Ca_LVAst','ik_SKv3_1','ik_SK_E2','na16.ina_ina','na16mut.ina_ina','na12.ina_ina','na12mut.ina_ina','i_pas'],
                #'currents'  : ['ihcn_Ih','na16.ina_ina','na16mut.ina_ina','na12.ina_ina','na12mut.ina_ina','i_pas'],
                # 'ionic_concentrations' :["cai", "ki", "nai"]
                'ionic_concentrations' :["ki", "nai"]
                }
#################################################################################

def modify_dict_file(filename, changes):
  """
  Modifies values in a dictionary stored in a text file.

  Args:
      filename: The name of the text file containing the dictionary.
      changes: A dictionary containing key-value pairs where the key is the key to modify in the original dictionary and the value is the new value.

  Raises:
      ValueError: If the file cannot be opened or the content is not valid JSON.
  """

  try:
    # Open the file and read its content
    with open(filename, "r") as file:
      content = file.read()

    # Try to load the content as a dictionary
    try:
      data = eval(content)  # Assuming the file contains valid dictionary syntax
    except (NameError, SyntaxError):
      raise ValueError("Invalid dictionary format in the file.")

    # Modify values based on the provided changes dictionary
    for key, value in changes.items():
      if key not in data:
        print(f"Warning: Key '{key}' not found in the dictionary, skipping.")
      else:
        data[key] = value

    # Write the modified dictionary back to the file
    # with open(filename, "w") as file:
    #   file.write(repr(data))
    with open(filename, "w") as file:
      file.write(json.dumps(data, indent=2))  # Add indentation for readability (optional)

  except IOError as e:
    raise ValueError(f"Error opening or writing file: {e}")
  

  #Don't forget to change NeuronModelClass.py to './Neuron_Model_12HH16HH/' and recompile!!


root_path_out = './Plots/GY_12HH16HH/Developing_2' ##path for saving your plots 
# Developing_2 : is the second method for creating the developing model: all na16 coefficients are zero, no nav1.6
# Developing: is the first method for creating the developing model: all na16 mechanisms are replaced with na12

if not os.path.exists(root_path_out): ##make directory if it doens't exist
        os.makedirs(root_path_out)


rbs_vshift = 13.5

filename12 = './params/na12_HH_082024.txt' ##12HH params file that you will update with values below in changesna12
filename16 = './params/na16_HH_081224.txt' ##16HH params file that you will update with values below in changesna16
filename_R850P = './params/R850P_HH_082024.txt'
## 12HH mod file params can be changed below, don't need it for 081224

config_dict = {"sim_config_soma": sim_config_soma,
              "sim_config_ais": sim_config_ais,
              "sim_config_basaldend": sim_config_basaldend,
              "sim_config_nexus": sim_config_nexus,
              "sim_config_apicaldend": sim_config_apicaldend}

config_dict2={"sim_config_nexus": sim_config_nexus,
              "sim_config_apicaldend": sim_config_apicaldend}

config_dict3={"sim_config_soma": sim_config_soma}

# for config_name, config in config_dict3.items():
path = f'20Aug_correctWT'

for factor in [0.1]:
  simwt = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=0,nav12=1,nav16=0, somaK=1*2.2, KP=25*0.15, KT=5,
                              ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1,soma_na12=3.2,node_na = 1,
                              na12name = 'na12_HH_082024',mut_name = 'na12_HH_082024',na12mechs = ['na12','na12mut'],
                              na16name =  'na12_HH_082024',na16mut_name =  'na12_HH_082024',na16mechs=['na12','na12mut'],params_folder = './params/',
                              plots_folder = f'{root_path_out}/{path}/', update=True, fac=None)
  wt_Vm1,_,wt_t1,_ = simwt.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = sim_config_soma)
  wt_fi=simwt.plot_fi_curve_2line(wt_data=None,wt2_data=None,start=0,end=2,nruns=100, fn=f'/WT-aisca_fac-{factor}')
  simwt.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx='WT')
  #NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities') ##TF uncomment to run function and plot channel densities in axon[0]
  
# ##het model
  sim_het = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=0,nav12=1,nav16=0, somaK=1*2.2, KP=25*0.15, KT=5,
                              ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=0,soma_na12=3.2,node_na = 1,
                              na12name = 'na12_HH_082024',mut_name = 'R850P_HH_082024',na12mechs = ['na12','na12mut'],
                              na16name =  'na12_HH_082024',na16mut_name = 'R850P_HH_082024',na16mechs=['na12','na12mut'],params_folder = './params/',
                              plots_folder = f'{root_path_out}/{path}/', update=True, fac=None)
  het_Vm1,_,het_t1,_ = sim_het.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = sim_config_soma) #sim_config for changing regions
  het_fi=sim_het.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=None,start=0,end=2,nruns=100, fn=f'WTHET-aisca-fac-{factor}')
  sim_het.wtvsmut_stim_dvdt(wt_Vm=wt_Vm1,wt_t=wt_t1,sim_config=sim_config_soma,vs_amp=[0.5], fnpre=f'WTvsHET')#sim_config for changing regions
  sim_het.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx='HET')
  # NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_Het') ##TF uncomment to run function and plot channel densities in axon[0]

  # ##KO model
  sim_ko = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=0,nav12=1,nav16=0, somaK=1*2.2, KP=25*0.15, KT=5,
                              ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=0,soma_na12=3.2,node_na = 1,
                              na12name = 'R850P_HH_082024',mut_name = 'R850P_HH_082024',na12mechs = ['na12','na12mut'],
                              na16name = 'R850P_HH_082024',na16mut_name = 'R850P_HH_082024',na16mechs=['na12','na12mut'],params_folder = './params/',
                              plots_folder = f'{root_path_out}/{path}/', update=True, fac=None)
  sim_ko.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=het_fi,start=0,end=2,nruns=100, fn=f'WTHETKO-aisca-fac-{factor}')
  sim_ko.wtvsmut_stim_dvdt(wt_Vm=wt_Vm1,wt_t=wt_t1,het_Vm=het_Vm1,het_t=het_t1,sim_config=sim_config_soma,vs_amp=[0.5], fnpre=f'WTvHETvKO_800sweep')#sim_config for changing regions
  sim_ko.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx='KO')
  #NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_KO') ##TF uncomment to run function and plot channel densities in axon[0]"""

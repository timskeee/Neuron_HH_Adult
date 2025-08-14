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

#################################################################################
#1
# sim_config_soma = {
                # 'section' : 'soma',
                # 'segment' : 0.5,
                # 'section_num': 0,
                # #'currents' : ['ina','ica','ik'],
                # 'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ihcn_Ih','ik_SK_E2','ik_SKv3_1'], #Somatic
                # #'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ik_SK_E2','ik_SKv3_1'], #AIS (no Ih)
                # #'currents'  : ['ica_Ca_HVA','ica_Ca_LVAst','ik_SKv3_1','ik_SK_E2','na16.ina_ina','na16mut.ina_ina','na12.ina_ina','na12mut.ina_ina','i_pas'],
                # #'currents'  : ['ihcn_Ih','ik_SKv3_1','na16.ina_ina','na16mut.ina_ina','na12.ina_ina','na12mut.ina_ina','i_pas'],
                # 'current_names' : ['Ih','SKv3_1','Na16 WT','Na16 WT','Na12','Na12 MUT','pas'],
                # 'ionic_concentrations' :["cai", "ki", "nai"]
                # #'ionic_concentrations' :["ki", "nai"]
                # }
# 2
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


root_path_out = './Plots/GY_12HH16HH/Nov/' ##path for saving your plots
if not os.path.exists(root_path_out): ##make directory if it doens't exist
        os.makedirs(root_path_out)


rbs_vshift = 13.5

filename12 = './params/na12annaTFHH2.txt' ##12HH params file that you will update with values below in changesna12
filename16 = './params/na16HH_TF2.txt' ##16HH params file that you will update with values below in changesna16
filenamemut = './params/na12annaTFHHmut.txt' 



## 1.2 HH params after re-fitting rbs_vshift 1.2. The rbs_vshift had weird inactivation curve that didn't bottom out at 0.
# changesna12={"Rd": 0.025712394696815438, "Rg": 0.01854277725353276, "Rb": 0.09013136340161398, "Ra": 0.3380714915775742, "a0s": 0.00036615946706607756, "gms": 0.14082624570054866, "hmin": 0.008420778920829085, "mmin": 0.013671131800210966, "qinf": 5.760329120353593, "q10": 2.289601426305275, "qg": 0.6693522946835427, "qd": 0.8058343822410788, "qa1": 5.835550042292994, "smax": 5.941545585888373, "sh": 8.886047186457889, "thinf": -40.114984963535186, "thi2": -77.41692349310195, "thi1": -60.488477521934875, "tha": -24.155451306086988, "vvs": 0.7672523706054653, "vvh": -53.184249317587984, "vhalfs": -33.73363659219147, "zetas": 13.419130866269455}

## 1.2HH params newest after Kevin's inpurt. 12-16 gap closer to 5mV now.
changesna12={"Rd": 0.023204006298533603, "Rg": 0.015604498120126004, "Rb": 0.0925081211054913, "Ra": 0.23933332265451177, "a0s": 0.0005226303768198727, "gms": 0.14418575154491814, "hmin": 0.008449935591049326, "mmin": 0.01193016441163175, "qinf": 5.7593653647578105, "q10": 2.1532859986639186, "qg": 1.2968193480468215, "qd": 0.661199851452832, "qa1": 4.492758160759386, "smax": 3.5557932199839737, "sh": 8.358558450280716, "thinf": -47.8194205612529, "thi2": -79.6556083820085, "thi1": -62.40165437813537, "tha": -33.850064879126805, "vvs": 1.4255479951467982, "vvh": -55.33213046147061, "vhalfs": -40.89976480829731, "zetas": 13.403615755952343}
# R850P Nov
changesna12_R850P = {"Rd": 0.020746368386756024, "Rg": 0.018111921274468024, "Rb": 0.027828201108899314, "Ra": 0.28523235200204017, "a0s": 0.0003110453447775763, "gms": 0.07128885357975331, "hmin": 0.006475560258124883, "mmin": 0.02347354198283438, "qinf": 6.75293487852935, "q10": 2.306874304104073, "qg": 1.3062676929883112, "qd": 0.7707059761076039, "qa1": 6.585879867084432, "smax": 3.4707353442204028, "sh": 9.27513934599908, "thinf": -52.06617357866996, "thi2": -79.22326533454797, "thi1": -60.277323827552806, "tha": -19.006578009999053, "vvs": 1.2884382787122886, "vvh": -41.32641030938734, "vhalfs": -34.36075663409899, "zetas": 8.33706211456386}
## 16HH mod file params can be changed below
changesna16 = {
        "sh": 8,
        "tha": -47+rbs_vshift,
        "qa": 7.2,
        "Ra": 0.4,
        "Rb": 0.124,
        "thi1": -61+rbs_vshift,
        "thi2": -61+rbs_vshift,
        "qd": 0.5,
        "qg": 1.5,
        "mmin": 0.02,  
        "hmin": 0.01,  
        "q10": 2,
        "Rg": 0.01,
        "Rd": 0.03,
        "thinf": -65+rbs_vshift,
        "qinf": 7,
        "vhalfs": -40+rbs_vshift,
        "a0s": 0.0003,
        "gms": 0.2,
        "zetas": 12,
        "smax": 10,
        "vvh": -58,
        "vvs": 2,
        "ar2": 1,
        #"ena": 55
        }

modify_dict_file(filename12, changesna12)
modify_dict_file(filenamemut, changesna12_R850P)
modify_dict_file(filename16, changesna16)


config_dict = {"sim_config_soma": sim_config_soma,
              "sim_config_ais": sim_config_ais,
              "sim_config_basaldend": sim_config_basaldend,
              "sim_config_nexus": sim_config_nexus,
              "sim_config_apicaldend": sim_config_apicaldend}

config_dict2={"sim_config_nexus": sim_config_nexus,
              "sim_config_apicaldend": sim_config_apicaldend}

config_dict3={"sim_config_soma": sim_config_soma}
# Mature
# For developing type 1 only replace the nav1.6 mod files with the nav1.2 mod files.
# For developing type 2 nav16=0, soma_na16=0, ais_nav16_factor=0
# For developing type 3 nav16=0, soma_na16=0, ais_nav16_factor=0, and nav12=2.3
# mature LOF is complete loss of function by removing one allele, S1655P

for config_name, config in config_dict3.items():
  path = f'Mature_LOF_S1655P'
  factor = 1

  ## WT vs HET vs Mut  
  ##WT model
  simwt = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=12*0.6,nav12=1,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,
                              ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                              na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                              na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                              plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  wt_Vm1,_,wt_t1,_ = simwt.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  wt_fi=simwt.plot_fi_curve_2line(wt_data=None,wt2_data=None,start=-0.4,end=1,nruns=50, fn=f'WT',epochlabel='200ms')
  simwt.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx='WT')
  #NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_WT') ##TF uncomment to run function and plot channel densities in axon[0]
  # features_df1 = ef.get_features(sim=simwt, prefix='WT', mut_name = 'na12annaTFHH2')
  """
  # ##het model
  sim_het = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=12*0.6,nav12=1,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,
                              ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                              na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHHmut',na12mechs = ['na12','na12mut'],
                              na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                              plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  het_Vm1,_,het_t1,_ = sim_het.get_stim_raw_data(stim_amp =0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #sim_config for changing regions
  #het_fi=sim_het.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=None,start=-0.4,end=1,nruns=50, fn=f'WTHET',epochlabel='200ms')
  sim_het.wtvsmut_stim_dvdt(wt_Vm=wt_Vm1,wt_t=wt_t1,sim_config=config,vs_amp=[0.5], fnpre=f'WTvsHET')#sim_config for changing regions
  sim_het.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx='HET')
  NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_Het') ##TF uncomment to run function and plot channel densities in axon[0]
  # features_df2 = ef.get_features(sim=sim_het, prefix='HET', mut_name = 'na12annaTFHH2')
  

  # ##KO model
  sim_ko = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=12*0.6,nav12=1,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,
                              ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                              na12name = 'na12annaTFHHmut',mut_name = 'na12annaTFHHmut',na12mechs = ['na12','na12mut'],
                              na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                              plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  #sim_ko.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=het_fi,start=-0.4,end=1,nruns=50, fn=f'WTHETKO',epochlabel='200ms')
  sim_ko.wtvsmut_stim_dvdt(wt_Vm=wt_Vm1,wt_t=wt_t1,het_Vm=het_Vm1,het_t=het_t1,sim_config=config,vs_amp=[0.5], fnpre=f'WTvHETvKO_ais16-{factor}')#vs_amp=[0.5]
  sim_ko.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx='KO')
  NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_KO') ##TF uncomment to run function and plot channel densities in axon[0]
  # features_df3 = ef.get_features(sim=sim_ko, prefix='KO', mut_name = 'na12annaTFHH2')
  """
 

 # for LOF S1655P
  factor = 0.5
  sim_het = tf.Na12Model_TF(
  ais_nav12_fac = factor*12,
  ais_nav16_fac = 12*0.6,
  nav12 = factor*1,nav16 = 1.3,
  somaK = 1*2.2*0.01, KP = 25*0.15, KT= 5,
  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,
  soma_na16=1*0.8,soma_na12=factor*3.2*0.8,
  node_na = 1,
  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)

  het_Vm1,_,het_t1,_ = sim_het.get_stim_raw_data(stim_amp =0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #sim_config for changing regions
  het_fi=sim_het.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=None,start=-0.4,end=1,nruns=50, fn=f'WTHET',epochlabel='200ms')
  sim_het.wtvsmut_stim_dvdt(wt_Vm=wt_Vm1,wt_t=wt_t1,sim_config=config,vs_amp=[0.5], fnpre=f'WTvsHET')#sim_config for changing regions
  sim_het.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx='HET')
  NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_Het') ##TF uncomment to run function and plot channel densities in axon[0]
  # features_df2 = ef.get_features(sim=sim_het, prefix='HET', mut_name = 'na12annaTFHH2')
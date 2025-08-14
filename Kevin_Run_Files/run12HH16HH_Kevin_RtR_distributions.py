from NeuronModelClass import NeuronModel
from NrnHelper import *
import NrnHelper as nh
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


root_path_out = './Plots/12HH16HH/12-CheckFI' ##path for saving your plots
if not os.path.exists(root_path_out): ##make directory if it doens't exist
        os.makedirs(root_path_out)




filename12 = './params/na12annaTFHH2.txt' ##12HH params file that you will update with values below in changesna12
filename16 = './params/na16HH_TF2.txt' ##16HH params file that you will update with values below in changesna16
filenamemut = './params/na12annaTFHHmut.txt' 



rbs_vshift = 13.5
## 1.2HH params newest after Kevin's inpurt. 12-16 gap closer to 5mV now.
changesna12={"Rd": 0.023204006298533603, "Rg": 0.015604498120126004, "Rb": 0.0925081211054913, "Ra": 0.23933332265451177, "a0s": 0.0005226303768198727, "gms": 0.14418575154491814, "hmin": 0.008449935591049326, "mmin": 0.01193016441163175, "qinf": 5.7593653647578105, "q10": 2.1532859986639186, "qg": 1.2968193480468215, "qd": 0.661199851452832, "qa1": 4.492758160759386, "smax": 3.5557932199839737, "sh": 8.358558450280716, "thinf": -47.8194205612529, "thi2": -79.6556083820085, "thi1": -62.40165437813537, "tha": -33.850064879126805, "vvs": 1.4255479951467982, "vvh": -55.33213046147061, "vhalfs": -40.89976480829731, "zetas": 13.403615755952343}
## 16HH mod file params can be changed below
changesna16 = {"Rd": 0.03, "Rg": 0.01, "Rb": 0.124, "Ra": 0.4, "a0s": 0.0003, "gms": 0.2, "hmin": 0.01, "mmin": 0.02, "qinf": 7, "q10": 2, "qg": 1.5, "qd": 0.5, "qa": 7.2, "smax": 10, "sh": 8, "thinf": -51.5, "thi2": -47.5, "thi1": -47.5, "tha": -33.5, "vvs": 2, "vvh": -58, "vhalfs": -26.5, "zetas": 12}
# changesna16 = {"sh": 8, "tha": -47+rbs_vshift, "qa": 7.2, "Ra": 0.4, "Rb": 0.124, "thi1": -61+rbs_vshift, "thi2": -61+rbs_vshift, "qd": 0.5, "qg": 1.5, "mmin": 0.02, "hmin": 0.01, "q10": 2, "Rg": 0.01, "Rd": 0.03, "thinf": -65+rbs_vshift, "qinf": 7, "vhalfs": -40+rbs_vshift, "a0s": 0.0003, "gms": 0.2, "zetas": 12, "smax": 10, "vvh": -58, "vvs": 2, "ar2": 1}

modify_dict_file(filename12, changesna12)
modify_dict_file(filename16, changesna16)


# config_dict = {"sim_config_soma": sim_config_soma,
#               "sim_config_ais": sim_config_ais,
#               "sim_config_basaldend": sim_config_basaldend,
#               "sim_config_nexus": sim_config_nexus,
#               "sim_config_apicaldend": sim_config_apicaldend}

# config_dict2={"sim_config_nexus": sim_config_nexus,
#               "sim_config_apicaldend": sim_config_apicaldend}

config_dict3={"sim_config_soma": sim_config_soma}

for config_name, config in config_dict3.items():
  path = f'2-checkFI_1350sweep'

allmutsefel = pd.DataFrame()
# fig,axs = plt.subplots(1,1)
# fig1, axs1 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# fig2, axs2 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# fig3, axs3 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# cmap = cm.get_cmap('rainbow')

# ratios1216 = {
#   'test':(1,1),
#   '100:0': (1, 0),
#   '90:10': (0.9, 0.1),
#   '80:20': (0.8, 0.2),
#   '70:30': (0.7, 0.3),
#   '60:40': (0.6, 0.4),
#   '50:50': (0.5, 0.5),
#   '40:60': (0.4, 0.6),
#   '30:70': (0.3, 0.7),
#   '20:80': (0.2, 0.8),
#   '10:90': (0.1, 0.9),
#   '0:100': (0, 1)
# }
# chantest ={'test':(1,4)}

# for key, (fac12, fac16) in chantest.items():

    #### THESE PARAMETERS ARE FOR THE NEW PARABOLIC AIS ONLY!!!!!!!!!!!! Changed KP *2 and ais_nav16_fac *0.5
# fac12=0.5
# fac16=2
# simdist = tf.Na12Model_TF(ais_nav12_fac=12*1.2*fac12,ais_nav16_fac=12*0.6*0.5*fac16,nav12=1,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15*2, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                   ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                   na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                   na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                   plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# NeuronModel.chandensities2(name = f'{root_path_out}/{path}/parabAIS_WT_5') ## includes soma/dendrites
# wt_Vm1,_,wt_t1,_ = simwt.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
    # wt_fi=simwt.plot_fi_curve_2line(wt_data=None,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'WT_FI', epochlabel='200ms')
    # # simwt.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx=f'WT-')
    # simwt.make_currentscape_plot(amp=0.5, time1=50,time2=300,stim_start=30, sweep_len=500,pfx=f'WT-')
  #   features_wt = ef.get_features(sim=simwt, prefix=f'{root_path_out}/{path}/WT', mut_name='WT')
  # allmutsefel = allmutsefel.append(features_wt, ignore_index=True)
# simwt.wtvsmut_stim_dvdt(wt_Vm=wt_Vm1,wt_t=wt_t1,sim_config=sim_config_soma,vs_amp=[0.5],stim_dur=200, fnpre=f'somaK-{fac12}__')



# Pretty solid set of factors.
# aisfac12=0.45
# aisfac16=0.35
# kpfac=1.5
# Kca=0.5
# ca=0.5

## AIS equal peak factors
fac12=0.5
fac16=2

## Best factors for new WT that fires 
aisfac12=0.4
aisfac16=0.3
kpfac=1.5
Kca=0.5
ca=0.5
na12=1.1
na16=1.1

## This set of params is for getting chandensity plot ONLY!
# sim_chandensities = tf.Na12Model_TF(ais_nav12_fac=12*1.2*fac12,ais_nav16_fac=12*0.6*0.5*fac16,nav12=1*na12,nav16=1.3*na16, somaK=1*2.2*0.01, KP=25*0.15*kpfac, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                     ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                     na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                     na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                     plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# NeuronModel.chandensities2(name = f'{root_path_out}/{path}/AIS_left_2') ## includes soma/dendrites
# input("Press Enter to continue...")


simwt = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12,ais_nav16_fac=12*0.6*0.5*aisfac16,nav12=1*na12,nav16=1.3*na16, somaK=1*2.2*0.01, KP=25*0.15*kpfac, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
                                    ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                    na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                    na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                    plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
wt_fi=simwt.plot_fi_curve_2line(wt_data=None,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'WT_FI', epochlabel='500ms')
# wt_Vm1,_,wt_t1,_ = simwt.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5


## Scanning KP to see if I can get more spikes with no dvdt change
# for fac in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:

#   ## adding *0.7 factor to KP increases spiking but doesn't change dvdt shape.
#   simmut = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12,ais_nav16_fac=12*0.6*0.5*aisfac16,nav12=1*na12,nav16=1.3*na16, somaK=1*2.2*0.01, KP=25*0.15*kpfac*fac, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                       ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                       na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                       na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                       plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
#   # wt_fi=simwt.plot_fi_curve_2line(wt_data=None,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'WT_FI', epochlabel='500ms')
#   simmut.wtvsmut_stim_dvdt(wt_Vm=wt_Vm1,wt_t=wt_t1,sim_config=sim_config_soma,vs_amp=[0.5],stim_dur=500, fnpre=f'KP-{fac}')



# fig4, axs4 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# fig5, axs5 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# cmap = cm.get_cmap('rainbow')
# # for i, factor in enumerate([18000,20000,21000,24000]):

# for i, factor in enumerate([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]):
#   fig4, axs4 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
#   fig5, axs5 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
#   cmap = cm.get_cmap('rainbow')
#   color = cmap(i/11)

#   simwt = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12*0.1,ais_nav16_fac=12*0.6*0.5*aisfac16*0.1,nav12=1*na12*0.1,nav16=1.3*na16*0.1, somaK=1*2.2*0.01, KP=25*0.15*kpfac, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                     ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                     na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                     na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                     plots_folder = f'{root_path_out}/{path}', update=True, fac=None)

#   # fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
#   # simwt.plot_stim(axs = axs[0],stim_amp = 0.5,dt=0.005,stim_dur=200, clr='cadetblue') #cadetblue
#   # plot_dvdt_from_volts(simwt.volt_soma, simwt.dt, axs[1],clr='cadetblue')
#   # fig_volts.savefig(f'{simwt.plot_folder}/parabAIS_5_ais12-{aisfac12}_ais16-{aisfac16}_KP-{kpfac}_Kca{Kca}_aisca-{ca}_12-{na12}_16-{na16}_stim0.5dvdt300.pdf') #Change output file path here
#   Vmwt,_,twt,_ = simwt.get_stim_raw_data(stim_amp =factor ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
#   dvdt4 = np.gradient(Vmwt)/0.005
#   axs4.plot(Vmwt[1:18000],dvdt4[1:18000],color=color, alpha=0.8,linewidth=1)
  
  
#   simwt2 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12*0,ais_nav16_fac=12*0.6*0.5*aisfac16*0,nav12=1*na12*0,nav16=1.3*na16*0, somaK=1*2.2*0.01, KP=25*0.15*kpfac, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                     ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                     na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                     na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                     plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
#   Vmwt2,_,twt2,_ = simwt2.get_stim_raw_data(stim_amp =factor ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
#   dvdt5 = np.gradient(Vmwt2)/0.005
#   axs4.plot(Vmwt2[1:18000],dvdt5[1:18000],color=color, alpha=0.8,linewidth=1)


#   out4 = f'{root_path_out}/{path}/1216-0.1_stim-{factor}.pdf'
#   fig4.savefig(out4)  
#   out5 = f'{root_path_out}/{path}/1216-0_stim-{factor}.pdf'
#   fig5.savefig(out5)  


# input("Press Enter to continue...")



# fig,axs = plt.subplots(1,1)
# fig1, axs1 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# fig2, axs2 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# fig3, axs3 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# cmap = cm.get_cmap('rainbow')

###############################################################################################################################################################
##### ais12
allmutsefel = pd.DataFrame()

ratios1216 = {
  'test':(1,1),
  # '100:0': (1, 0),
  # '90:10': (0.9, 0.1),
  # '80:20': (0.8, 0.2),
  # '70:30': (0.7, 0.3),
  # '60:40': (0.6, 0.4),
  # '50:50': (0.5, 0.5),
  # '40:60': (0.4, 0.6),
  # '30:70': (0.3, 0.7),
  # '20:80': (0.2, 0.8),
  # '10:90': (0.1, 0.9),
  # '0:100': (0, 1)
}

ratios1216flip={
  'WT':(1,1),
  'het':(0.5,1),
  'ko':(0,1),
# '0:100': (0, 1),
# '10:90': (0.1, 0.9),
# '20:80': (0.2, 0.8),
# '30:70': (0.3, 0.7),
# '40:60': (0.4, 0.6),
# '50:50': (0.5, 0.5),
# '60:40': (0.6, 0.4),
# '70:30': (0.7, 0.3),
# '80:20': (0.8, 0.2),
# '90:10': (0.9, 0.1),
# '100:0': (1, 0)
}

ratios1216_double={
  # 'WT':(1,1),
'0:100': (0, 2),
'10:90': (0.2, 1.8),
'20:80': (0.4, 1.6),
'30:70': (0.6, 1.4),
'40:60': (0.8, 1.2),
'50:50': (1.0, 1.0),
'60:40': (1.2, 0.8),
'70:30': (1.4, 0.6),
'80:20': (1.6, 0.4),
'90:10': (1.8, 0.2),
'100:0': (2, 0)
}
for key, (factor12, factor16) in ratios1216flip.items():

# for i, factor in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
# for i, factor12 in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
#   for i, factor16 in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
# for i, factor in enumerate([1,0.9]):
  # color = cmap(i/11)
     
  ## varying 1.2 and 1.6 with different AIS distributions for paper heatmaps.
  sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12*factor12,ais_nav16_fac=12*0.6*0.5*aisfac16*factor16,nav12=1*na12*factor12,nav16=1.3*na16*factor16, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
                              ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                              na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                              na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                              plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  sim1216.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'1-{key}-FI',epochlabel='200ms')

  # fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
  # sim1216.plot_stim(axs = axs[0],stim_amp = 0.5,dt=0.005,stim_dur=1700, clr='cadetblue') #cadetblue
  # plot_dvdt_from_volts(sim1216.volt_soma, sim1216.dt, axs[1],clr='cadetblue')
  # fig_volts.savefig(f'{sim1216.plot_folder}/{key}_rerun.pdf') #Change output file path here
  # features_wt = ef.get_features(sim=sim1216, prefix=f'{root_path_out}/{path}/{key}_1216_left2_stim0.5', mut_name=f'{root_path_out}/{path}/left2_{key}')
  # allmutsefel = allmutsefel.append(features_wt, ignore_index=True)
# allmutsefel.to_csv(f'{root_path_out}/{path}/EFEL_shiftAIS_right5_.csv')







## 1.2 factor
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12*factor,ais_nav16_fac=12*0.6*0.5*aisfac16,nav12=1*na12*factor,nav16=1.3*na16, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
  #                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                                 na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  #                                 na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
  # dvdt1 = np.gradient(Vm12)/0.005
  # axs1.plot(Vm12[1:12000],dvdt1[1:12000],color=color, alpha=0.8,linewidth=1)


## 1.6 factor
  # sim16 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12,ais_nav16_fac=12*0.6*0.5*aisfac16*factor,nav12=1*na12,nav16=1.3*na16*factor, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
  #                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                                 na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  #                                 na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.5 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
  # dvdt2 = np.gradient(Vm16)/0.005
  # axs2.plot(Vm16[1:14000],dvdt2[1:14000],color=color, alpha=0.8,linewidth=1)

  # sim16.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'16-{factor}-FI',epochlabel='200ms')
  # sim16.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx=f'16-{factor}-')

  # fig_volts,axs5 = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
  # sim16.plot_stim(axs = axs5[0],stim_amp = 0.3,dt=0.005, clr=color)
  # plot_dvdt_from_volts(sim16.volt_soma, sim12.dt, axs5[1],clr=color)
  # fig_volts.savefig(f'{sim16.plot_folder}/16-{factor}-spikedvdt.pdf') #Change output file path here 


##1.2 and 1.6 factor
  # sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12*factor,ais_nav16_fac=12*0.6*0.5*aisfac16*factor,nav12=1*na12*factor,nav16=1.3*na16*factor, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
  #                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                                 na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  #                                 na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm1216,_,t1216,_ = sim1216.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
  # dvdt3 = np.gradient(Vm1216)/0.005
  # axs3.plot(Vm1216[1:15000],dvdt3[1:15000],color=color, alpha=0.8,linewidth=1)

  # sim1216.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'1216-{factor}-FI')
  # sim1216.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx=f'1216-{factor}-')

  # fig_volts,axs5 = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
  # sim1216.plot_stim(axs = axs5[0],stim_amp = 0.3,dt=0.005, clr=color)
  # plot_dvdt_from_volts(sim1216.volt_soma, sim12.dt, axs5[1],clr=color)
  # fig_volts.savefig(f'{sim1216.plot_folder}/1216-{factor}-spikedvdt.pdf') #Change output file path here  
    
  # color = cmap(stim/11)
  # axs.plot(Vm[8000:14000],dvdt[8000:14000],color=color, alpha=0.8,linewidth=1)






# ##1.6 factor Adding 0% 1.6 with bigger stim to get something on 1216 plot
# sim16 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12,ais_nav16_fac=12*0.6*0.5*aisfac16*0.2,nav12=1*na12,nav16=1.3*na16*0.2, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.7 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt5 = np.gradient(Vm16)/0.005
# color = cmap(8/11)
# axs2.plot(Vm16[1:12000],dvdt5[1:12000],color=color, alpha=0.8,linewidth=1)

# sim16 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12,ais_nav16_fac=12*0.6*0.5*aisfac16*0.1,nav12=1*na12,nav16=1.3*na16*0.1, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.9 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt5 = np.gradient(Vm16)/0.005
# color = cmap(9/11)
# axs2.plot(Vm16[1:10000],dvdt5[1:10000],color=color, alpha=0.8,linewidth=1)

# sim16 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12,ais_nav16_fac=12*0.6*0.5*aisfac16*0,nav12=1*na12,nav16=1.3*na16*0, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =1.5 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt5 = np.gradient(Vm16)/0.005
# color = cmap(10/11)
# axs2.plot(Vm16[1:9000],dvdt5[1:9000],color=color, alpha=0.8,linewidth=1)





# ##1.2 & 1.6 factor Adding 0% 1.6 with bigger stim to get something on 1216 plot
# sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12*0.3,ais_nav16_fac=12*0.6*0.5*aisfac16*0.3,nav12=1*na12*0.3,nav16=1.3*na16*0.3, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim1216.get_stim_raw_data(stim_amp =0.6 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt4 = np.gradient(Vm16)/0.005
# color = cmap(7/11)
# axs3.plot(Vm16[1:18000],dvdt4[1:18000],color=color, alpha=0.8,linewidth=1)

# sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12*0.2,ais_nav16_fac=12*0.6*0.5*aisfac16*0.2,nav12=1*na12*0.2,nav16=1.3*na16*0.2, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim1216.get_stim_raw_data(stim_amp =0.75 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt4 = np.gradient(Vm16)/0.005
# color = cmap(8/11)
# axs3.plot(Vm16[1:18000],dvdt4[1:18000],color=color, alpha=0.8,linewidth=1)

# sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12*0.1,ais_nav16_fac=12*0.6*0.5*aisfac16*0.1,nav12=1*na12*0.01,nav16=1.3*na16*0.1, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim1216.get_stim_raw_data(stim_amp =1.4 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt4 = np.gradient(Vm16)/0.005
# color = cmap(9/11)
# axs3.plot(Vm16[1:18000],dvdt4[1:18000],color=color, alpha=0.8,linewidth=1)

# sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*1.2*aisfac12*0,ais_nav16_fac=12*0.6*0.5*aisfac16*0,nav12=1*na12*0,nav16=1.3*na16*0, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim1216.get_stim_raw_data(stim_amp =1.2 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt4 = np.gradient(Vm16)/0.005
# color = cmap(10/11)
# axs3.plot(Vm16[1:18000],dvdt4[1:18000],color=color, alpha=0.8,linewidth=1)



# suf= f'022125'
# # out1 = f'{root_path_out}/{path}/dvdt12-{suf}.pdf'
# # out2 = f'{root_path_out}/{path}/dvdt16-{suf}.pdf'
# out3 = f'{root_path_out}/{path}/dvdt1216-{suf}.pdf'
# # axs.legend()  # Add a legend to the plot
# # fig1.savefig(out1)
# # fig2.savefig(out2)
# fig3.savefig(out3)

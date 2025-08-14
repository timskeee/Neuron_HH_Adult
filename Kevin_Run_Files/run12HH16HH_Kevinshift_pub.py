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


root_path_out = './Plots/12HH16HH/5-newAIS_raiseDVDT' ##path for saving your plots
if not os.path.exists(root_path_out): ##make directory if it doens't exist
        os.makedirs(root_path_out)


rbs_vshift = 13.5

filename12 = './params/na12annaTFHH.txt' ##12HH params file that you will update with values below in changesna12
filename16 = './params/na16HH_TF2.txt' ##16HH params file that you will update with values below in changesna16

## 12HH mod file params can be changed below
# changesna12 = {
#         "sh": 8,
#         "tha": -38+rbs_vshift,
#         "qa": 5.41,
#         "Ra": 0.3282,
#         "Rb": 0.1,
#         "thi1": -80+rbs_vshift,#-80,
#         "thi2": -80+rbs_vshift,#-80,
#         "qd": 0.5,
#         "qg": 1.5,
#         "mmin": 0.02,
#         "hmin": 0.01,
#         "Rg": 0.01,
#         "Rd": 0.02657,
#         "thinf": -53+rbs_vshift,
#         "qinf": 7.69,
#         "vhalfs": -60+rbs_vshift,
#         "a0s": 0.0003,
#         "gms": 0.2,
#         "q10": 2,
#         "zetas": 12,
#         "smax": 10,
#         "vvh": -58,
#         "vvs": 2,
#         "ar2": 1,
#         "Ena": 55,
#         }

## 1.2 HH params after re-fitting rbs_vshift 1.2. The rbs_vshift had weird inactivation curve that didn't bottom out at 0.
changesna12={"Rd": 0.025712394696815438, 
             "Rg": 0.01854277725353276, 
             "Rb": 0.09013136340161398, 
             "Ra": 0.3380714915775742, 
             "a0s": 0.00036615946706607756, 
             "gms": 0.14082624570054866, 
             "hmin": 0.008420778920829085, 
             "mmin": 0.013671131800210966, 
             "qinf": 5.760329120353593, 
             "q10": 2.289601426305275, 
             "qg": 0.6693522946835427, 
             "qd": 0.8058343822410788, 
             "qa1": 5.835550042292994, 
             "smax": 5.941545585888373, 
             "sh": 8.886047186457889, 
             "thinf": -40.114984963535186, 
             "thi2": -77.41692349310195, 
             "thi1": -60.488477521934875, 
             "tha": -24.155451306086988, 
             "vvs": 0.7672523706054653, 
             "vvh": -53.184249317587984, 
             "vhalfs": -33.73363659219147, 
             "zetas": 13.419130866269455}

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
modify_dict_file(filename16, changesna16)


config_dict3={"sim_config_soma": sim_config_soma}

for config_name, config in config_dict3.items():
  path = f'64-allpubfigs_2000swp_allFIs'


# for factor in [0.00001,0.0001,0.001,0.01,0.1,0.25,0.5,0.75,1,1.2,2,4,6,10,25,50,100,1000,10000,100000]:
# for amps in np.arange(-0.4,0.4,0.05):


# for factor in [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:

# fig,axs = plt.subplots(1,1)
# fig1, axs1 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# fig2, axs2 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# fig3, axs3 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))

# xmin=-80
# xmax=40
# ymin=-100
# ymax=700

# xmin1=0
# xmax1=2
# ymin1=-80
# ymax1=40


# axs1.set_xlim(xmin, xmax)  # Replace xmin and xmax with your desired limits
# axs1.set_ylim(ymin, ymax)  # Replace ymin and ymax with your desired limits
# axs1.set_xticks([-80,-60,-40,-20,0,20,40])
# axs1.set_yticks([-100,0,100,200,300,400,500,600])

# axs2.set_xlim(xmin, xmax)
# axs2.set_ylim(ymin, ymax)
# axs2.set_xticks([-80,-60,-40,-20,0,20,40])
# axs2.set_yticks([-100,0,100,200,300,400,500,600])
                
# axs3.set_xlim(xmin, xmax)
# axs3.set_ylim(ymin, ymax)
# axs3.set_xticks([-80,-60,-40,-20,0,20,40])
# axs3.set_yticks([-100,0,100,200,300,400,500,600])                

cmap = cm.get_cmap('rainbow')
# for factor2 in [0.0001,0.001,0.01,0.1,0.5,1,2,4,6]:
# for factor2 in [1]:

shift5={"mut5":{"Rd": 0.023204006298533603, "Rg": 0.015604498120126004, "Rb": 0.0925081211054913, "Ra": 0.23933332265451177, "a0s": 0.0005226303768198727, "gms": 0.14418575154491814, "hmin": 0.008449935591049326, "mmin": 0.01193016441163175, "qinf": 5.7593653647578105, "q10": 2.1532859986639186, "qg": 1.2968193480468215, "qd": 0.661199851452832, "qa1": 4.492758160759386, "smax": 3.5557932199839737, "sh": 8.358558450280716, "thinf": -47.8194205612529, "thi2": -79.6556083820085, "thi1": -62.40165437813537, "tha": -33.850064879126805, "vvs": 1.4255479951467982, "vvh": -55.33213046147061, "vhalfs": -40.89976480829731, "zetas": 13.403615755952343}}


for mutname,dict in shift5.items():
  print(f"mutname is {mutname}")
  print(f"it's corresponding dictionary is {dict}")
  modify_dict_file(filename12,dict)
#   # modify_dict_file(filename16,dict)
  simwt = tf.Na12Model_TF(ais_nav12_fac=12*1.2,ais_nav16_fac=12*0.6,nav12=1,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  wt_Vm,_,wt_t,_ = simwt.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=1600, sim_config = config) #stim_amp=0.5
  wt_fi=simwt.plot_fi_curve_2line(wt_data=None,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'WT_FI', epochlabel='1600ms')
  
  for fac in [1.2]:
    ###############################################################################################################################################################
    ##### ais12
    # for i, factor in enumerate([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
    for i, factor in enumerate([0.4,0.3,0.2,0.1,0]):
    # for i, factor in enumerate([0]):
      color = cmap(i/11)
    
      ## 1.2 factor
      sim12 = tf.Na12Model_TF(ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.6,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.75,dt=0.005,rec_extra=False,stim_dur=1600, sim_config = config) #stim_amp=0.5
      # dvdt1 = np.gradient(Vm12)/0.005
      # axs1.plot(Vm12[1:12000],dvdt1[1:12000],color=color, alpha=0.8,linewidth=1)
      
      # NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_{factor}')

      sim12.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'12-{factor}-FI',epochlabel='1600ms')
      # sim12.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx=f'12-{factor}-')

      # fig_volts,axs4 = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
      
      # sim12.plot_stim(axs = axs4[0],stim_amp = 0.3,dt=0.005, clr=color, stim_dur = 1600)
      # plot_dvdt_from_volts(sim12.volt_soma, sim12.dt, axs4[1],clr=color)
      # axs4[0].set_xlim(xmin1, xmax1)
      # axs4[0].set_ylim(ymin1, ymax1)
      # axs4[0].set_xticks([0.0, 0.5,1,1.5,2])
      # axs4[0].set_yticks([-80,-60,-40,-20,0,20,40])
      # axs4[1].set_xlim(xmin, xmax)
      # axs4[1].set_ylim(ymin, ymax)
      # axs4[1].set_xticks([-80,-60,-40,-20,0,20,40])
      # axs4[1].set_yticks([-100,0,100,200,300,400,500,600])
      # fig_volts.savefig(f'{sim12.plot_folder}/12-{factor}-spikedvdt.pdf') #Change output file path here 
      



      ##1.6 factor
      sim16 = tf.Na12Model_TF(ais_nav12_fac=12*fac,ais_nav16_fac=12*0.6*factor,nav12=1,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5, #ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.5 ,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      # dvdt2 = np.gradient(Vm16)/0.005
      # axs2.plot(Vm16[1:15000],dvdt2[1:15000],color=color, alpha=0.8,linewidth=1)

      sim16.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'16-{factor}-FI',epochlabel='1600ms')
      # sim16.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx=f'16-{factor}-')

      # fig_volts,axs5 = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
      # sim16.plot_stim(axs = axs5[0],stim_amp = 0.3,dt=0.005, clr=color, stim_dur = 1600)
      # plot_dvdt_from_volts(sim16.volt_soma, sim12.dt, axs5[1],clr=color)
      # axs5[0].set_xlim(xmin1, xmax1)
      # axs5[0].set_ylim(ymin1, ymax1)
      # axs5[0].set_xticks([0.0, 0.5,1,1.5,2])
      # axs5[0].set_yticks([-80,-60,-40,-20,0,20,40])
      # axs5[1].set_xlim(xmin, xmax)
      # axs5[1].set_ylim(ymin, ymax)
      # axs5[1].set_xticks([-80,-60,-40,-20,0,20,40])
      # axs5[1].set_yticks([-100,0,100,200,300,400,500,600])
      # fig_volts.savefig(f'{sim16.plot_folder}/16-{factor}-spikedvdt.pdf') #Change output file path here 




      ##1.2 and 1.6 factor
      sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.6*factor,nav12=1*factor,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8*factor,soma_na12=3.2*0.8*factor,node_na = 1,
                                  na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm1216,_,t1216,_ = sim1216.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      # dvdt3 = np.gradient(Vm1216)/0.005
      # axs3.plot(Vm1216[1:15000],dvdt3[1:15000],color=color, alpha=0.8,linewidth=1)

      sim1216.plot_fi_curve_2line(wt_data=wt_fi,wt2_data=None,start=-0.4,end=1,nruns=140, fn=f'1216-{factor}-FI',epochlabel='1600ms')
      # sim1216.make_currentscape_plot(amp=0.5, time1=50,time2=100,stim_start=30, sweep_len=200,pfx=f'1216-{factor}-')

      # fig_volts,axs6 = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
      # sim1216.plot_stim(axs = axs6[0],stim_amp = 0.3,dt=0.005, clr=color, stim_dur = 1600)
      # plot_dvdt_from_volts(sim1216.volt_soma, sim12.dt, axs6[1],clr=color)
      # axs6[0].set_xlim(xmin1, xmax1)
      # axs6[0].set_ylim(ymin1, ymax1)
      # axs6[0].set_xticks([0.0, 0.5,1,1.5,2])
      # axs6[0].set_yticks([-80,-60,-40,-20,0,20,40])
      # axs6[1].set_xlim(xmin, xmax)
      # axs6[1].set_ylim(ymin, ymax)
      # axs6[1].set_xticks([-80,-60,-40,-20,0,20,40])
      # axs6[1].set_yticks([-100,0,100,200,300,400,500,600])
      # fig_volts.savefig(f'{sim1216.plot_folder}/1216-{factor}-spikedvdt.pdf') #Change output file path here  
      
      # color = cmap(stim/11)
      # axs.plot(Vm[8000:14000],dvdt[8000:14000],color=color, alpha=0.8,linewidth=1)


    # ##1.6 factor Adding 0% 1.6 with bigger stim to get something
    # sim16 = tf.Na12Model_TF(ais_nav12_fac=12*fac,ais_nav16_fac=12*0.6*0,nav12=1,nav16=1.3*0, somaK=1*2.2*0.01, KP=25*0.15, KT=5, #ais_nav16_fac=12*0.2*factor
    #                             ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
    #                             na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
    #                             na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
    #                             plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
    # Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =1.5 ,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
    # dvdt2 = np.gradient(Vm16)/0.005
    # axs2.plot(Vm16[1:15000],dvdt2[1:15000],color=color, alpha=0.8,linewidth=1)



    # suf= f'{mutname}_FIXED_AXES'
    # out1 = f'{root_path_out}/{path}/dvdt12-{suf}.pdf'
    # out2 = f'{root_path_out}/{path}/dvdt16-{suf}.pdf'
    # out3 = f'{root_path_out}/{path}/dvdt1216-{suf}.pdf'
    # # axs.legend()  # Add a legend to the plot
    # fig1.savefig(out1)
    # fig2.savefig(out2)
    # fig3.savefig(out3)

    
  









  ##############################################################################################################################
  ## 44-vshift12_092424
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5

  ## 45-vshift12_092424 only muts 5-7
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.2,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  
  
  ## 46-vshift12_092424 only muts 3-7
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.5,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  
  
  ## 47-vshift12_092424 only muts 3-5
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.4,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  
  
  ## 48-vshift12_092424 only muts 3-5 **Original AIS 0-ais12/3
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.4,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  
  
  ## 49-vshift12_092424 only 7. updated AIS. 
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.2,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  
  ## 50-vshift12_092424 only 7. Playing with AIS. Adding more fraction of 1.6.
  
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.2,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH',mut_name = 'na12annaTFHH',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5

  ## 51-vshift12_092424 Completely reworking AIS. 


  ##AIS as of 092524.
  # ////////////////////////////////////////
	# //New distribution from Kevin's guidance 092424//
	# //"have 1.2 drop to 0 gBar by segment 3 instead of segment 5.  i know we're going for Hu Shu style where it looks like half half but the data support something more like 1/3rd 1.2
	# //then take the 1.6 value at 6 and move that to position 3.  connect with a smooth line" - Kevin
	
	# //WT 1.6
	# gbar_na16(0:2*ais_end/10) = 0:ais_na16/3
	# gbar_na16(2*ais_end/10:4*ais_end/10) = ais_na16/3:ais_na16/2
	# gbar_na16(4*ais_end/10:8*ais_end/10) = ais_na16/2:ais_na16/2
	# gbar_na16(8*ais_end/10:9*ais_end/10) = ais_na16/2:ais_na16/3
	# gbar_na16(9*ais_end/10:ais_end) = ais_na16/3:naked_axon_na/2
	
	# //Mut 1.6
	# gbar_na16mut(0:2*ais_end/10) = 0:ais_na16/3
	# gbar_na16mut(2*ais_end/10:4*ais_end/10) = ais_na16/3:ais_na16/2
	# gbar_na16mut(4*ais_end/10:8*ais_end/10) = ais_na16/2:ais_na16/2
	# gbar_na16mut(8*ais_end/10:9*ais_end/10) = ais_na16/2:ais_na16/3
	# gbar_na16mut(9*ais_end/10:ais_end) = ais_na16/3:naked_axon_na/2

	# gbar_na16(ais_end:1) = naked_axon_na/2:naked_axon_na/2 // 1/5th nav1.6
	# gbar_na16mut(ais_end:1) = naked_axon_na/2:naked_axon_na/2 // 1/5th nav1.6


	# //WT 1.2
	# //gbar_na12(0:ais_end/10) = 0:ais_na12/3
	# gbar_na12(0:ais_end/10) = ais_na12/3:ais_na12/2
	# gbar_na12(ais_end/10:2*ais_end/10) = ais_na12/2:ais_na12/2
	# gbar_na12(2*ais_end/10:3*ais_end/10) = ais_na12/2:ais_na12/2
	# //gbar_na12(3*ais_end/10:4*ais_end/10) = ais_na12/2:ais_na12/6
	# gbar_na12(3*ais_end/10:4*ais_end/10) = ais_na12/2:0
	# gbar_na12(4*ais_end/10:ais_end) = 0:0 //##TF071624 changed to 0

	# //Mut 1.2
	# //gbar_na12mut(0:ais_end/10) = 0:ais_na12/3
	# gbar_na12mut(0:ais_end/10) = ais_na12/3:ais_na12/2
	# gbar_na12mut(ais_end/10:2*ais_end/10) = ais_na12/2:ais_na12/2
	# gbar_na12mut(2*ais_end/10:3*ais_end/10) = ais_na12/2:ais_na12/2
	# //gbar_na12mut(3*ais_end/10:4*ais_end/10) = ais_na12/2:ais_na12/6
	# gbar_na12mut(3*ais_end/10:4*ais_end/10) = ais_na12/2:0
	# gbar_na12mut(4*ais_end/10:ais_end) = 0:0 //##TF071624 changed to 0
	
	# gbar_na12(ais_end:1) = 0:0 //naked axon ##Don't start naked axon until end of AIS
	# gbar_na12mut(ais_end:1) = 0:0 //naked axon ##Don't start naked axon until end of AIS
	# ////////////////////////////////////////
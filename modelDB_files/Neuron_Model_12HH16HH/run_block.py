from NeuronModelClass import NeuronModel
from NrnHelper import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Na12HH import *
import Na12HH as tf
import os
import efel_feature_extractor as ef
import pandas as pd


sim_config_soma = {
                'section' : 'soma',
                'segment' : 0.5,
                'section_num': 0,
                'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ihcn_Ih','ik_SK_E2','ik_SKv3_1'],
                'current_names' : ['Ih','SKv3_1','Na16 WT','Na16 WT','Na12','Na12 MUT','pas'],
                'ionic_concentrations' :["cai", "ki", "nai"]
                }

def modify_dict_file(filename, changes):
  try:
    with open(filename, "r") as file:
      content = file.read()

    try:
      data = eval(content)  
    except (NameError, SyntaxError):
      raise ValueError("Invalid dictionary format in the file.")

    for key, value in changes.items():
      if key not in data:
        print(f"Warning: Key '{key}' not found in the dictionary, skipping.")
      else:
        data[key] = value

    with open(filename, "w") as file:
      file.write(json.dumps(data, indent=2))

  except IOError as e:
    raise ValueError(f"Error opening or writing file: {e}")
  

root_path_out = './Plots' ##path for saving your plots
if not os.path.exists(root_path_out): ##make directory if it doens't exist
        os.makedirs(root_path_out)
filename12 = './params/na12WT.txt' 
filename16 = './params/na16WT.txt'
changesna12={"Rd": 0.023204006298533603, "Rg": 0.015604498120126004, "Rb": 0.0925081211054913, "Ra": 0.23933332265451177, "a0s": 0.0005226303768198727, "gms": 0.14418575154491814, "hmin": 0.008449935591049326, "mmin": 0.01193016441163175, "qinf": 5.7593653647578105, "q10": 2.1532859986639186, "qg": 1.2968193480468215, "qd": 0.661199851452832, "qa1": 4.492758160759386, "smax": 3.5557932199839737, "sh": 8.358558450280716, "thinf": -47.8194205612529, "thi2": -79.6556083820085, "thi1": -62.40165437813537, "tha": -33.850064879126805, "vvs": 1.4255479951467982, "vvh": -55.33213046147061, "vhalfs": -40.89976480829731, "zetas": 13.403615755952343}
changesna16 = {"Rd": 0.03, "Rg": 0.01, "Rb": 0.124, "Ra": 0.4, "a0s": 0.0003, "gms": 0.2, "hmin": 0.01, "mmin": 0.02, "qinf": 7, "q10": 2, "qg": 1.5, "qd": 0.5, "qa": 7.2, "smax": 10, "sh": 8, "thinf": -51.5, "thi2": -47.5, "thi1": -47.5, "tha": -33.5, "vvs": 2, "vvh": -58, "vhalfs": -26.5, "zetas": 12}
modify_dict_file(filename12, changesna12)
modify_dict_file(filename16, changesna16)
config_dict3={"sim_config_soma": sim_config_soma}

for config_name, config in config_dict3.items():
  path = f'Plots'

allmutsefel = pd.DataFrame()
aisfac12=0.4
aisfac16=0.3
kpfac=1.5
Kca=0.5
ca=0.5
na12=1.1
na16=1.1

# fig,axs = plt.subplots(1,1)
# fig1, axs1 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# fig2, axs2 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# fig3, axs3 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
# cmap = cm.get_cmap('rainbow')

ratios1216flip={
'WT':(1,1),
'0:100': (0, 1),
'10:90': (0.1, 0.9),
'20:80': (0.2, 0.8),
'30:70': (0.3, 0.7),
'40:60': (0.4, 0.6),
'50:50': (0.5, 0.5),
'60:40': (0.6, 0.4),
'70:30': (0.7, 0.3),
'80:20': (0.8, 0.2),
'90:10': (0.9, 0.1),
'100:0': (1, 0)
}
for key, (factor12, factor16) in ratios1216flip.items():

# for i, factor12 in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
#   for i, factor16 in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
# for i, factor in enumerate([1,0.9]):
  # color = cmap(i/11)
     
  sim1216 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12*factor12,ais_nav16_fac=12*0.6*0.5*aisfac16*factor16,nav12=1*na12*factor12,nav16=1.3*na16*factor16, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
                              ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                              na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
                              na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
                              plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
  sim1216.plot_stim(axs = axs[0],stim_amp = 0.5,dt=0.005,stim_dur=1700, clr='cadetblue') #cadetblue
  plot_dvdt_from_volts(sim1216.volt_soma, sim1216.dt, axs[1],clr='cadetblue')
  fig_volts.savefig(f'{sim1216.plot_folder}/{key}_WT.pdf') #Change output file path here
  # features_wt = ef.get_features(sim=sim1216, prefix=f'{root_path_out}/{path}/{key}_1216_right5_stim0.5', mut_name=f'{root_path_out}/{path}/right5_{key}')
  # allmutsefel = allmutsefel.append(features_wt, ignore_index=True)

## 1.2 factor
  # sim12 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12*factor,ais_nav16_fac=12*0.6*0.5*aisfac16,nav12=1*na12*factor,nav16=1.3*na16, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
  #                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                                 na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
  #                                 na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
  # dvdt1 = np.gradient(Vm12)/0.005
  # axs1.plot(Vm12[1:12000],dvdt1[1:12000],color=color, alpha=0.8,linewidth=1)


## 1.6 factor
  # sim16 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12,ais_nav16_fac=12*0.6*0.5*aisfac16*factor,nav12=1*na12,nav16=1.3*na16*factor, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
  #                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                                 na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
  #                                 na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.5 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
  # dvdt2 = np.gradient(Vm16)/0.005
  # axs2.plot(Vm16[1:14000],dvdt2[1:14000],color=color, alpha=0.8,linewidth=1)


##1.2 and 1.6 factor
  # sim1216 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12*factor,ais_nav16_fac=12*0.6*0.5*aisfac16*factor,nav12=1*na12*factor,nav16=1.3*na16*factor, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
  #                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                                 na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
  #                                 na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm1216,_,t1216,_ = sim1216.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
  # dvdt3 = np.gradient(Vm1216)/0.005
  # axs3.plot(Vm1216[1:15000],dvdt3[1:15000],color=color, alpha=0.8,linewidth=1)


    
  # color = cmap(stim/11)
  # axs.plot(Vm[8000:14000],dvdt[8000:14000],color=color, alpha=0.8,linewidth=1)


# sim16 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12,ais_nav16_fac=12*0.6*0.5*aisfac16*0.2,nav12=1*na12,nav16=1.3*na16*0.2, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.7 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt5 = np.gradient(Vm16)/0.005
# color = cmap(8/11)
# axs2.plot(Vm16[1:12000],dvdt5[1:12000],color=color, alpha=0.8,linewidth=1)

# sim16 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12,ais_nav16_fac=12*0.6*0.5*aisfac16*0.1,nav12=1*na12,nav16=1.3*na16*0.1, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.9 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt5 = np.gradient(Vm16)/0.005
# color = cmap(9/11)
# axs2.plot(Vm16[1:10000],dvdt5[1:10000],color=color, alpha=0.8,linewidth=1)

# sim16 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12,ais_nav16_fac=12*0.6*0.5*aisfac16*0,nav12=1*na12,nav16=1.3*na16*0, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =1.5 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt5 = np.gradient(Vm16)/0.005
# color = cmap(10/11)
# axs2.plot(Vm16[1:9000],dvdt5[1:9000],color=color, alpha=0.8,linewidth=1)


# sim1216 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12*0.3,ais_nav16_fac=12*0.6*0.5*aisfac16*0.3,nav12=1*na12*0.3,nav16=1.3*na16*0.3, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim1216.get_stim_raw_data(stim_amp =0.6 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt4 = np.gradient(Vm16)/0.005
# color = cmap(7/11)
# axs3.plot(Vm16[1:18000],dvdt4[1:18000],color=color, alpha=0.8,linewidth=1)

# sim1216 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12*0.2,ais_nav16_fac=12*0.6*0.5*aisfac16*0.2,nav12=1*na12*0.2,nav16=1.3*na16*0.2, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim1216.get_stim_raw_data(stim_amp =0.75 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt4 = np.gradient(Vm16)/0.005
# color = cmap(8/11)
# axs3.plot(Vm16[1:18000],dvdt4[1:18000],color=color, alpha=0.8,linewidth=1)

# sim1216 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12*0.1,ais_nav16_fac=12*0.6*0.5*aisfac16*0.1,nav12=1*na12*0.01,nav16=1.3*na16*0.1, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim1216.get_stim_raw_data(stim_amp =1.4 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt4 = np.gradient(Vm16)/0.005
# color = cmap(9/11)
# axs3.plot(Vm16[1:18000],dvdt4[1:18000],color=color, alpha=0.8,linewidth=1)

# sim1216 = tf.Na12HH(ais_nav12_fac=12*1.2*aisfac12*0,ais_nav16_fac=12*0.6*0.5*aisfac16*0,nav12=1*na12*0,nav16=1.3*na16*0, somaK=1*2.2*0.01, KP=25*0.15*kpfac*0.7, KT=5,#ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*0.75
#                                 ais_ca = 100*8.6*0.1*ca,ais_Kca = 0.5*Kca,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
#                                 na12name = 'na12WT',mut_name = 'na12WT',na12mechs = ['na12','na12mut'],
#                                 na16name = 'na16WT',na16mut_name = 'na16WT',na16mechs=['na16','na16mut'],params_folder = './params/',
#                                 plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
# Vm16,_,t16,_ = sim1216.get_stim_raw_data(stim_amp =1.2 ,dt=0.005,rec_extra=False,stim_dur=200, sim_config = config) #stim_amp=0.5
# dvdt4 = np.gradient(Vm16)/0.005
# color = cmap(10/11)
# axs3.plot(Vm16[1:18000],dvdt4[1:18000],color=color, alpha=0.8,linewidth=1)

# suf= f'0225'
# # out1 = f'{root_path_out}/{path}/dvdt12-{suf}.pdf'
# # out2 = f'{root_path_out}/{path}/dvdt16-{suf}.pdf'
# out3 = f'{root_path_out}/{path}/dvdt1216-{suf}.pdf'
# # axs.legend()  # Add a legend to the plot
# # fig1.savefig(out1)
# # fig2.savefig(out2)
# fig3.savefig(out3)

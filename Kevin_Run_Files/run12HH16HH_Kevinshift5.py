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

filename12 = './params/na12annaTFHH2.txt' ##12HH params file that you will update with values below in changesna12
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
  path = f'52-shift5_scans_092524'


# for factor in [0.00001,0.0001,0.001,0.01,0.1,0.25,0.5,0.75,1,1.2,2,4,6,10,25,50,100,1000,10000,100000]:
# for amps in np.arange(-0.4,0.4,0.05):


# for factor in [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:

# fig,axs = plt.subplots(1,1)
fig1, axs1 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
fig2, axs2 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
fig3, axs3 = plt.subplots(figsize=(cm_to_in(8), cm_to_in(8)))
cmap = cm.get_cmap('rainbow')
# for factor2 in [0.0001,0.001,0.01,0.1,0.5,1,2,4,6]:
# for factor2 in [1]:



shifts ={"2-mut24_1":{"Rd": 0.022029682875262854, "Rg": 0.018120063782468342, "Rb": 0.10321991286929143, "Ra": 0.32604109671371134, "a0s": 0.0002604300165281561, "gms": 0.09781162218506306, "hmin": 0.013883807967526541, "mmin": 0.01212397133763213, "qinf": 5.891781829699699, "q10": 2.2688991798289626, "qg": 0.6473601462066132, "qd": 0.8050125081337014, "qa1": 4.722532397419606, "smax": 4.03736924200435, "sh": 9.892667771598681, "thinf": -44.00119027292933, "thi2": -78.45833451534541, "thi1": -67.99027549994665, "tha": -30.551456725040797, "vvs": 0.9210563782024217, "vvh": -52.617456996608915, "vhalfs": -35.3499711791991, "zetas": 13.774743505054236},
"2-mut24_2":{"Rd": 0.023168163893342877, "Rg": 0.01600356086938636, "Rb": 0.10726974834537298, "Ra": 0.23957720924998074, "a0s": 0.0003391247060452528, "gms": 0.148475769372005, "hmin": 0.001543036546658038, "mmin": 0.011127268172722474, "qinf": 5.020907948455523, "q10": 2.2938215084295512, "qg": 1.0001574744525517, "qd": 0.9048939582826149, "qa1": 4.428356719494654, "smax": 7.231259618000946, "sh": 8.135408672899468, "thinf": -45.080771693034684, "thi2": -76.19575131575387, "thi1": -60.48752712495384, "tha": -31.021032204629368, "vvs": 0.5102471209520576, "vvh": -53.597094328000615, "vhalfs": -51.12443179628696, "zetas": 13.894637953993424},
"2-mut24_3":{"Rd": 0.02278301686336839, "Rg": 0.01692809850731326, "Rb": 0.13918729996221382, "Ra": 0.22894043591196978, "a0s": 0.000372234410805627, "gms": 0.1415939813468497, "hmin": 0.010550781122172673, "mmin": 0.011202064427311422, "qinf": 5.087645832279604, "q10": 2.6705331165039268, "qg": 0.01058233666262065, "qd": 0.797713909864653, "qa1": 6.3032811169180185, "smax": 8.988136282878056, "sh": 8.89912802644549, "thinf": -45.66857889247376, "thi2": -77.41692349310195, "thi1": -59.71422330600601, "tha": -33.68656751789582, "vvs": 0.2257608735705567, "vvh": -48.2750382801924, "vhalfs": -28.960886783524384, "zetas": 8.954200273293168},
"2-mut24_4":{"Rd": 0.02684013117872433, "Rg": 0.01859212372238241, "Rb": 0.0958871396067906, "Ra": 0.19565342697707, "a0s": 0.00018497137189466583, "gms": 0.1484405949663394, "hmin": 0.00819368563597951, "mmin": 0.004049816571217133, "qinf": 5.272616357906921, "q10": 1.9892861649438314, "qg": 0.8561687112859035, "qd": 0.9500554192687916, "qa1": 4.573074019706725, "smax": 1.7975141899319236, "sh": 7.300727314446856, "thinf": -46.11333270045915, "thi2": -69.4275275975793, "thi1": -59.67316938187846, "tha": -33.22974482220169, "vvs": 1.1652456838853915, "vvh": -51.79011731149271, "vhalfs": -42.391066511335154, "zetas": 13.563294107855276},
"2-mut24_5":{"Rd": 0.023204006298533603, "Rg": 0.015604498120126004, "Rb": 0.0925081211054913, "Ra": 0.23933332265451177, "a0s": 0.0005226303768198727, "gms": 0.14418575154491814, "hmin": 0.008449935591049326, "mmin": 0.01193016441163175, "qinf": 5.7593653647578105, "q10": 2.1532859986639186, "qg": 1.2968193480468215, "qd": 0.661199851452832, "qa1": 4.492758160759386, "smax": 3.5557932199839737, "sh": 8.358558450280716, "thinf": -47.8194205612529, "thi2": -79.6556083820085, "thi1": -62.40165437813537, "tha": -33.850064879126805, "vvs": 1.4255479951467982, "vvh": -55.33213046147061, "vhalfs": -40.89976480829731, "zetas": 13.403615755952343},
"2-mut24_6":{"Rd": 0.02583088545989115, "Rg": 0.01977254074886893, "Rb": 0.08040862241968169, "Ra": 0.17963869744085534, "a0s": 0.00045051652231592275, "gms": 0.14910952525648863, "hmin": 0.007044302741707565, "mmin": 0.002755536342927248, "qinf": 6.2199921242226, "q10": 2.3339523418942, "qg": 1.0226410680839544, "qd": 0.7479248328154102, "qa1": 5.5247255856732025, "smax": 7.498128866141444, "sh": 9.644646128625473, "thinf": -48.95283407942315, "thi2": -78.01876225129938, "thi1": -63.606554795013345, "tha": -38.124623422337805, "vvs": 2.0202480783271097, "vvh": -58.02143839108442, "vhalfs": -53.138339358987125, "zetas": 14.205571494059567},
"2-mut24_7":{"Rd": 0.027040655367536696, "Rg": 0.01788843167887597, "Rb": 0.10691038625905988, "Ra": 0.2817003937800848, "a0s": 0.0003768308985175273, "gms": 0.03997006808431528, "hmin": 0.008685061713842906, "mmin": 0.013671131800210966, "qinf": 7.146246876661122, "q10": 2.6555196676949326, "qg": 1.163860705058872, "qd": 0.8916767166343843, "qa1": 6.682116443205098, "smax": 1.7189656404114961, "sh": 8.31064175557174, "thinf": -50.84698844259533, "thi2": -78.4191760154983, "thi1": -56.75375047747005, "tha": -35.11519744465838, "vvs": 0.1677937465231482, "vvh": -49.836820367560605, "vhalfs": -13.723820074612954, "zetas": 13.512305037269085}}

shifts345={"2-mut24_3":{"Rd": 0.02278301686336839, "Rg": 0.01692809850731326, "Rb": 0.13918729996221382, "Ra": 0.22894043591196978, "a0s": 0.000372234410805627, "gms": 0.1415939813468497, "hmin": 0.010550781122172673, "mmin": 0.011202064427311422, "qinf": 5.087645832279604, "q10": 2.6705331165039268, "qg": 0.01058233666262065, "qd": 0.797713909864653, "qa1": 6.3032811169180185, "smax": 8.988136282878056, "sh": 8.89912802644549, "thinf": -45.66857889247376, "thi2": -77.41692349310195, "thi1": -59.71422330600601, "tha": -33.68656751789582, "vvs": 0.2257608735705567, "vvh": -48.2750382801924, "vhalfs": -28.960886783524384, "zetas": 8.954200273293168},
"2-mut24_4":{"Rd": 0.02684013117872433, "Rg": 0.01859212372238241, "Rb": 0.0958871396067906, "Ra": 0.19565342697707, "a0s": 0.00018497137189466583, "gms": 0.1484405949663394, "hmin": 0.00819368563597951, "mmin": 0.004049816571217133, "qinf": 5.272616357906921, "q10": 1.9892861649438314, "qg": 0.8561687112859035, "qd": 0.9500554192687916, "qa1": 4.573074019706725, "smax": 1.7975141899319236, "sh": 7.300727314446856, "thinf": -46.11333270045915, "thi2": -69.4275275975793, "thi1": -59.67316938187846, "tha": -33.22974482220169, "vvs": 1.1652456838853915, "vvh": -51.79011731149271, "vhalfs": -42.391066511335154, "zetas": 13.563294107855276},
"2-mut24_5":{"Rd": 0.023204006298533603, "Rg": 0.015604498120126004, "Rb": 0.0925081211054913, "Ra": 0.23933332265451177, "a0s": 0.0005226303768198727, "gms": 0.14418575154491814, "hmin": 0.008449935591049326, "mmin": 0.01193016441163175, "qinf": 5.7593653647578105, "q10": 2.1532859986639186, "qg": 1.2968193480468215, "qd": 0.661199851452832, "qa1": 4.492758160759386, "smax": 3.5557932199839737, "sh": 8.358558450280716, "thinf": -47.8194205612529, "thi2": -79.6556083820085, "thi1": -62.40165437813537, "tha": -33.850064879126805, "vvs": 1.4255479951467982, "vvh": -55.33213046147061, "vhalfs": -40.89976480829731, "zetas": 13.403615755952343},}

shifts567={"2-mut24_5":{"Rd": 0.023204006298533603, "Rg": 0.015604498120126004, "Rb": 0.0925081211054913, "Ra": 0.23933332265451177, "a0s": 0.0005226303768198727, "gms": 0.14418575154491814, "hmin": 0.008449935591049326, "mmin": 0.01193016441163175, "qinf": 5.7593653647578105, "q10": 2.1532859986639186, "qg": 1.2968193480468215, "qd": 0.661199851452832, "qa1": 4.492758160759386, "smax": 3.5557932199839737, "sh": 8.358558450280716, "thinf": -47.8194205612529, "thi2": -79.6556083820085, "thi1": -62.40165437813537, "tha": -33.850064879126805, "vvs": 1.4255479951467982, "vvh": -55.33213046147061, "vhalfs": -40.89976480829731, "zetas": 13.403615755952343},
"2-mut24_6":{"Rd": 0.02583088545989115, "Rg": 0.01977254074886893, "Rb": 0.08040862241968169, "Ra": 0.17963869744085534, "a0s": 0.00045051652231592275, "gms": 0.14910952525648863, "hmin": 0.007044302741707565, "mmin": 0.002755536342927248, "qinf": 6.2199921242226, "q10": 2.3339523418942, "qg": 1.0226410680839544, "qd": 0.7479248328154102, "qa1": 5.5247255856732025, "smax": 7.498128866141444, "sh": 9.644646128625473, "thinf": -48.95283407942315, "thi2": -78.01876225129938, "thi1": -63.606554795013345, "tha": -38.124623422337805, "vvs": 2.0202480783271097, "vvh": -58.02143839108442, "vhalfs": -53.138339358987125, "zetas": 14.205571494059567},
"2-mut24_7":{"Rd": 0.027040655367536696, "Rg": 0.01788843167887597, "Rb": 0.10691038625905988, "Ra": 0.2817003937800848, "a0s": 0.0003768308985175273, "gms": 0.03997006808431528, "hmin": 0.008685061713842906, "mmin": 0.013671131800210966, "qinf": 7.146246876661122, "q10": 2.6555196676949326, "qg": 1.163860705058872, "qd": 0.8916767166343843, "qa1": 6.682116443205098, "smax": 1.7189656404114961, "sh": 8.31064175557174, "thinf": -50.84698844259533, "thi2": -78.4191760154983, "thi1": -56.75375047747005, "tha": -35.11519744465838, "vvs": 0.1677937465231482, "vvh": -49.836820367560605, "vhalfs": -13.723820074612954, "zetas": 13.512305037269085}}

shift5={"mut5":{"Rd": 0.023204006298533603, "Rg": 0.015604498120126004, "Rb": 0.0925081211054913, "Ra": 0.23933332265451177, "a0s": 0.0005226303768198727, "gms": 0.14418575154491814, "hmin": 0.008449935591049326, "mmin": 0.01193016441163175, "qinf": 5.7593653647578105, "q10": 2.1532859986639186, "qg": 1.2968193480468215, "qd": 0.661199851452832, "qa1": 4.492758160759386, "smax": 3.5557932199839737, "sh": 8.358558450280716, "thinf": -47.8194205612529, "thi2": -79.6556083820085, "thi1": -62.40165437813537, "tha": -33.850064879126805, "vvs": 1.4255479951467982, "vvh": -55.33213046147061, "vhalfs": -40.89976480829731, "zetas": 13.403615755952343}}

shift7 = {"2-mut24_7":{"Rd": 0.027040655367536696, "Rg": 0.01788843167887597, "Rb": 0.10691038625905988, "Ra": 0.2817003937800848, "a0s": 0.0003768308985175273, "gms": 0.03997006808431528, "hmin": 0.008685061713842906, "mmin": 0.013671131800210966, "qinf": 7.146246876661122, "q10": 2.6555196676949326, "qg": 1.163860705058872, "qd": 0.8916767166343843, "qa1": 6.682116443205098, "smax": 1.7189656404114961, "sh": 8.31064175557174, "thinf": -50.84698844259533, "thi2": -78.4191760154983, "thi1": -56.75375047747005, "tha": -35.11519744465838, "vvs": 0.1677937465231482, "vvh": -49.836820367560605, "vhalfs": -13.723820074612954, "zetas": 13.512305037269085}}

for mutname,dict in shift5.items():
  print(f"mutname is {mutname}")
  print(f"it's corresponding dictionary is {dict}")
  modify_dict_file(filename12,dict)
#   # modify_dict_file(filename16,dict)

  for fac in [1.5,1.2,0.9,0.5,0.25]:
    ###############################################################################################################################################################
    ##### ais12
    for i, factor in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
      color = cmap(i/11)
    
      ## 1.2 factor
      sim12 = tf.Na12Model_TF(ais_nav12_fac=12*fac*factor,ais_nav16_fac=12,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt1 = np.gradient(Vm12)/0.005
      axs1.plot(Vm12[1:15000],dvdt1[1:15000],color=color, alpha=0.8,linewidth=1)
      # NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_ais16-1-{factor}')
      
      ##1.6 factor
      sim16 = tf.Na12Model_TF(ais_nav12_fac=12*fac,ais_nav16_fac=12*factor,nav12=1,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5, #ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.5 ,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt2 = np.gradient(Vm16)/0.005
      axs2.plot(Vm16[1:15000],dvdt2[1:15000],color=color, alpha=0.8,linewidth=1)

      ##1.2 and 1.6 factor
      sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*fac*factor,ais_nav16_fac=12*factor,nav12=1*factor,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8*factor,soma_na12=3.2*0.8*factor,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm1216,_,t1216,_ = sim1216.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt3 = np.gradient(Vm1216)/0.005
      axs3.plot(Vm1216[1:15000],dvdt3[1:15000],color=color, alpha=0.8,linewidth=1) 
      
      # color = cmap(stim/11)
      # axs.plot(Vm[8000:14000],dvdt[8000:14000],color=color, alpha=0.8,linewidth=1)
      
    suf= f'{mutname}_ais12-{fac}'
    out1 = f'{root_path_out}/{path}/dvdt12-{suf}.pdf'
    out2 = f'{root_path_out}/{path}/dvdt16-{suf}.pdf'
    out3 = f'{root_path_out}/{path}/dvdt1216-{suf}.pdf'
    # axs.legend()  # Add a legend to the plot
    fig1.savefig(out1)
    fig2.savefig(out2)
    fig3.savefig(out3)

    ###############################################################################################################################################################
    ##### ais16
    for i, factor in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
      color = cmap(i/11)
    
      ## 1.2 factor
      sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*fac,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt1 = np.gradient(Vm12)/0.005
      axs1.plot(Vm12[1:15000],dvdt1[1:15000],color=color, alpha=0.8,linewidth=1)
      # NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_ais16-1-{factor}')
      
      ##1.6 factor
      sim16 = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=12*fac*factor,nav12=1,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5, #ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.5 ,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt2 = np.gradient(Vm16)/0.005
      axs2.plot(Vm16[1:15000],dvdt2[1:15000],color=color, alpha=0.8,linewidth=1)

      ##1.2 and 1.6 factor
      sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*fac*factor,nav12=1*factor,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8*factor,soma_na12=3.2*0.8*factor,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm1216,_,t1216,_ = sim1216.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt3 = np.gradient(Vm1216)/0.005
      axs3.plot(Vm1216[1:15000],dvdt3[1:15000],color=color, alpha=0.8,linewidth=1) 
      
      # color = cmap(stim/11)
      # axs.plot(Vm[8000:14000],dvdt[8000:14000],color=color, alpha=0.8,linewidth=1)
      
    suf= f'{mutname}_ais16-{fac}'
    out1 = f'{root_path_out}/{path}/dvdt12-{suf}.pdf'
    out2 = f'{root_path_out}/{path}/dvdt16-{suf}.pdf'
    out3 = f'{root_path_out}/{path}/dvdt1216-{suf}.pdf'
    # axs.legend()  # Add a legend to the plot
    fig1.savefig(out1)
    fig2.savefig(out2)
    fig3.savefig(out3)

    ###############################################################################################################################################################
    ##### nav12
    for i, factor in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
      color = cmap(i/11)
    
      ## 1.2 factor
      sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12,nav12=1*fac*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt1 = np.gradient(Vm12)/0.005
      axs1.plot(Vm12[1:15000],dvdt1[1:15000],color=color, alpha=0.8,linewidth=1)
      # NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_ais16-1-{factor}')
      
      ##1.6 factor
      sim16 = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=12*factor,nav12=1*fac,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5, #ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.5 ,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt2 = np.gradient(Vm16)/0.005
      axs2.plot(Vm16[1:15000],dvdt2[1:15000],color=color, alpha=0.8,linewidth=1)

      ##1.2 and 1.6 factor
      sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*factor,nav12=1*fac*factor,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8*factor,soma_na12=3.2*0.8*factor,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm1216,_,t1216,_ = sim1216.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt3 = np.gradient(Vm1216)/0.005
      axs3.plot(Vm1216[1:15000],dvdt3[1:15000],color=color, alpha=0.8,linewidth=1) 
      
      # color = cmap(stim/11)
      # axs.plot(Vm[8000:14000],dvdt[8000:14000],color=color, alpha=0.8,linewidth=1)
      
    suf= f'{mutname}_nav12-{fac}'
    out1 = f'{root_path_out}/{path}/dvdt12-{suf}.pdf'
    out2 = f'{root_path_out}/{path}/dvdt16-{suf}.pdf'
    out3 = f'{root_path_out}/{path}/dvdt1216-{suf}.pdf'
    # axs.legend()  # Add a legend to the plot
    fig1.savefig(out1)
    fig2.savefig(out2)
    fig3.savefig(out3)

    ###############################################################################################################################################################
    ##### nav16
    for i, factor in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
      color = cmap(i/11)
    
      ## 1.2 factor
      sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12,nav12=1*factor,nav16=1.3*fac, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt1 = np.gradient(Vm12)/0.005
      axs1.plot(Vm12[1:15000],dvdt1[1:15000],color=color, alpha=0.8,linewidth=1)
      # NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_ais16-1-{factor}')
      
      ##1.6 factor
      sim16 = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=12*factor,nav12=1,nav16=1.3*fac*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5, #ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.5 ,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt2 = np.gradient(Vm16)/0.005
      axs2.plot(Vm16[1:15000],dvdt2[1:15000],color=color, alpha=0.8,linewidth=1)

      ##1.2 and 1.6 factor
      sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*factor,nav12=1*factor,nav16=1.3*fac*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8*factor,soma_na12=3.2*0.8*factor,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm1216,_,t1216,_ = sim1216.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt3 = np.gradient(Vm1216)/0.005
      axs3.plot(Vm1216[1:15000],dvdt3[1:15000],color=color, alpha=0.8,linewidth=1) 
      
      # color = cmap(stim/11)
      # axs.plot(Vm[8000:14000],dvdt[8000:14000],color=color, alpha=0.8,linewidth=1)
      
    suf= f'{mutname}_nav16-{fac}'
    out1 = f'{root_path_out}/{path}/dvdt12-{suf}.pdf'
    out2 = f'{root_path_out}/{path}/dvdt16-{suf}.pdf'
    out3 = f'{root_path_out}/{path}/dvdt1216-{suf}.pdf'
    # axs.legend()  # Add a legend to the plot
    fig1.savefig(out1)
    fig2.savefig(out2)
    fig3.savefig(out3)

    ###############################################################################################################################################################
    ##### soma_na16
    for i, factor in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
      color = cmap(i/11)
    
      ## 1.2 factor
      sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8*fac,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt1 = np.gradient(Vm12)/0.005
      axs1.plot(Vm12[1:15000],dvdt1[1:15000],color=color, alpha=0.8,linewidth=1)
      # NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_ais16-1-{factor}')
      
      ##1.6 factor
      sim16 = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=12*factor,nav12=1,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5, #ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8*fac,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.5 ,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt2 = np.gradient(Vm16)/0.005
      axs2.plot(Vm16[1:15000],dvdt2[1:15000],color=color, alpha=0.8,linewidth=1)

      ##1.2 and 1.6 factor
      sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*factor,nav12=1*factor,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8*fac*factor,soma_na12=3.2*0.8*factor,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm1216,_,t1216,_ = sim1216.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt3 = np.gradient(Vm1216)/0.005
      axs3.plot(Vm1216[1:15000],dvdt3[1:15000],color=color, alpha=0.8,linewidth=1) 
      
      # color = cmap(stim/11)
      # axs.plot(Vm[8000:14000],dvdt[8000:14000],color=color, alpha=0.8,linewidth=1)
      
    suf= f'{mutname}_soma16-{fac}'
    out1 = f'{root_path_out}/{path}/dvdt12-{suf}.pdf'
    out2 = f'{root_path_out}/{path}/dvdt16-{suf}.pdf'
    out3 = f'{root_path_out}/{path}/dvdt1216-{suf}.pdf'
    # axs.legend()  # Add a legend to the plot
    fig1.savefig(out1)
    fig2.savefig(out2)
    fig3.savefig(out3)

    ###############################################################################################################################################################
    ##### soma_na12
    for i, factor in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
      color = cmap(i/11)
    
      ## 1.2 factor
      sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8*fac,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt1 = np.gradient(Vm12)/0.005
      axs1.plot(Vm12[1:15000],dvdt1[1:15000],color=color, alpha=0.8,linewidth=1)
      # NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_ais16-1-{factor}')
      
      ##1.6 factor
      sim16 = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=12*factor,nav12=1,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5, #ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8*fac,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.5 ,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt2 = np.gradient(Vm16)/0.005
      axs2.plot(Vm16[1:15000],dvdt2[1:15000],color=color, alpha=0.8,linewidth=1)

      ##1.2 and 1.6 factor
      sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*factor,nav12=1*factor,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8*factor,soma_na12=3.2*0.8*fac*factor,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm1216,_,t1216,_ = sim1216.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt3 = np.gradient(Vm1216)/0.005
      axs3.plot(Vm1216[1:15000],dvdt3[1:15000],color=color, alpha=0.8,linewidth=1) 
      
      # color = cmap(stim/11)
      # axs.plot(Vm[8000:14000],dvdt[8000:14000],color=color, alpha=0.8,linewidth=1)
      
    suf= f'{mutname}_soma12-{fac}'
    out1 = f'{root_path_out}/{path}/dvdt12-{suf}.pdf'
    out2 = f'{root_path_out}/{path}/dvdt16-{suf}.pdf'
    out3 = f'{root_path_out}/{path}/dvdt1216-{suf}.pdf'
    # axs.legend()  # Add a legend to the plot
    fig1.savefig(out1)
    fig2.savefig(out2)
    fig3.savefig(out3)

    ###############################################################################################################################################################
    ##### KP
    for i, factor in enumerate([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]):
      color = cmap(i/11)
    
      ## 1.2 factor
      sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15*fac, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt1 = np.gradient(Vm12)/0.005
      axs1.plot(Vm12[1:15000],dvdt1[1:15000],color=color, alpha=0.8,linewidth=1)
      # NeuronModel.chandensities(name = f'{root_path_out}/{path}/densities_ais16-1-{factor}')
      
      ##1.6 factor
      sim16 = tf.Na12Model_TF(ais_nav12_fac=12,ais_nav16_fac=12*factor,nav12=1,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15*fac, KT=5, #ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm16,_,t16,_ = sim16.get_stim_raw_data(stim_amp =0.5 ,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt2 = np.gradient(Vm16)/0.005
      axs2.plot(Vm16[1:15000],dvdt2[1:15000],color=color, alpha=0.8,linewidth=1)

      ##1.2 and 1.6 factor
      sim1216 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*factor,nav12=1*factor,nav16=1.3*factor, somaK=1*2.2*0.01, KP=25*0.15*fac, KT=5,#ais_nav16_fac=12*0.2*factor
                                  ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8*factor,soma_na12=3.2*0.8*factor,node_na = 1,
                                  na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
                                  na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
                                  plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
      Vm1216,_,t1216,_ = sim1216.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
      dvdt3 = np.gradient(Vm1216)/0.005
      axs3.plot(Vm1216[1:15000],dvdt3[1:15000],color=color, alpha=0.8,linewidth=1) 
      
      # color = cmap(stim/11)
      # axs.plot(Vm[8000:14000],dvdt[8000:14000],color=color, alpha=0.8,linewidth=1)
      
    suf= f'{mutname}_KP-{fac}'
    out1 = f'{root_path_out}/{path}/dvdt12-{suf}.pdf'
    out2 = f'{root_path_out}/{path}/dvdt16-{suf}.pdf'
    out3 = f'{root_path_out}/{path}/dvdt1216-{suf}.pdf'
    # axs.legend()  # Add a legend to the plot
    fig1.savefig(out1)
    fig2.savefig(out2)
    fig3.savefig(out3)

  









  ##############################################################################################################################
  ## 44-vshift12_092424
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5

  ## 45-vshift12_092424 only muts 5-7
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.2,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  
  
  ## 46-vshift12_092424 only muts 3-7
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.5,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  
  
  ## 47-vshift12_092424 only muts 3-5
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.4,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  
  
  ## 48-vshift12_092424 only muts 3-5 **Original AIS 0-ais12/3
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.4,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  
  
  ## 49-vshift12_092424 only 7. updated AIS. 
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.2,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5
  
  ## 50-vshift12_092424 only 7. Playing with AIS. Adding more fraction of 1.6.
  
  # sim12 = tf.Na12Model_TF(ais_nav12_fac=12*factor,ais_nav16_fac=12*0.2,nav12=1*factor,nav16=1.3, somaK=1*2.2*0.01, KP=25*0.15, KT=5,#ais_nav16_fac=12*0.2*factor
  #                               ais_ca = 100*8.6*0.1,ais_Kca = 0.5,soma_na16=1*0.8,soma_na12=3.2*0.8,node_na = 1,
  #                               na12name = 'na12annaTFHH2',mut_name = 'na12annaTFHH2',na12mechs = ['na12','na12mut'],
  #                               na16name = 'na16HH_TF2',na16mut_name = 'na16HH_TF2',na16mechs=['na16','na16mut'],params_folder = './params/',
  #                               plots_folder = f'{root_path_out}/{path}', update=True, fac=None)
  # Vm12,_,t12,_ = sim12.get_stim_raw_data(stim_amp = 0.5,dt=0.005,rec_extra=False,stim_dur=500, sim_config = config) #stim_amp=0.5

  ## 51-vshift12_092424 Completely reworking AIS. 
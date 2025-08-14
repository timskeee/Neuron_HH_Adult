from Na12HMMModel_TF import *
import matplotlib.pyplot as plt
import numpy as np
from neuron import h
import efel
efel.api.setDoubleSetting('Threshold', -40) 
import pandas as pd
import math
from scipy.signal import find_peaks

def get_sim_volt_values(sim,mut_name,rec_extra = False,dt = 0.005,stim_amp = 0.5): 
    sim.dt= dt
    rec_extra = True
    sim.l5mdl.init_stim(amp=stim_amp)
    if rec_extra:
        Vm, I, t, stim,extra_vms = sim.l5mdl.run_model(dt=dt,rec_extra = rec_extra)

        sim.extra_vms = extra_vms
    else:
        Vm, I, t, stim = sim.l5mdl.run_model(dt=dt)

        extra_vms = {}

    return Vm,t,extra_vms,I,stim


def get_sim_volt_valuesTF(sim,mut_name,dt = 0.005,stim_amp = 0.5): 
   
    sim.dt= dt
    #sim.make_het()
    rec_extra = False
    sim.l5mdl.init_stim(amp=stim_amp)
    if rec_extra:
        Vm, I, t, stim,_ = sim.l5mdl.run_sim_model(dt=dt)

        # sim.extra_vms = extra_vms
    else:
        Vm, I, t, stim,_ = sim.l5mdl.run_sim_model(dt=dt)

        # extra_vms = {}

    return Vm,t,I,stim


def get_features(sim,prefix=None,mut_name = 'na12WT',rec_extra=True):
    print("running routine")
    dt=0.005#0.1#0.005
    Vm,t,extra_vms,_,__ = get_sim_volt_values(sim,mut_name,rec_extra=rec_extra)
    stim_start = 200 #100 original
    stim_end = 1900 #800 original
    trace={}
    trace = {'T':t,'V':Vm,'stim_start':[stim_start],'stim_end':[stim_end]}
    trace['T']= trace['T'] * 1000
    #for neu
    feature_list= ['AP_height','AP_width','AP1_peak','AP1_width','Spikecount','all_ISI_values'] 
    
    traces = [trace]
    features = efel.getFeatureValues(traces,feature_list)
    

    try:
        features[0]['ISI mean'] =features[0]['all_ISI_values'].mean()
    except Exception as e:
        features[0]['ISI mean'] = 0
    features[0]['AP_height'] =features[0]['AP_height'].mean()
    features[0]['AP_width'] =features[0]['AP_width'][0].mean()
    features[0]['AP1_peak'] =features[0]['AP1_peak'][0]
    features[0]['AP1_width'] =features[0]['AP1_width'][0]
    features[0]['Spikecount'] =features[0]['Spikecount'][0]
    spike_count= features[0]['Spikecount']
    isi_values = features[0]['all_ISI_values']
    median_spike = int(math.floor(spike_count/2)) + 1
    start = int((stim_start + isi_values[0:median_spike-1].sum())/dt) 
    start2 = int((stim_start + isi_values[0:5].sum())/dt) 
    start3 = int((stim_start+isi_values[1])/dt) 
    start4 = int((stim_start+isi_values[0:3].sum())/dt) 
    try:
        end = 380000
    except Exception as e:
        end=400000
        print("There were not enough spikes to calculate median isi")

    volt_segment = Vm[start4:start2] 
    dvdt = np.gradient(volt_segment) / dt 
    filtered_indices = np.where(dvdt > 50)[0] 
    filtered_dvdt = dvdt[filtered_indices] 
    filtered_volt_segment = volt_segment[filtered_indices] 
    dvdtslope = np.diff(filtered_dvdt) 
    peaks, peaks_vals = find_peaks(filtered_dvdt) 

    negative_slope_indices = []
    negative_slopes = []

    n=5
    for peak in peaks:
        start_index = peak + n 
        if start_index < len(dvdtslope):            
            segment_indices = np.where(dvdtslope[start_index:] < 0)[0] + start_index + 1 
            segment_indices = segment_indices[segment_indices < len(filtered_dvdt)]
            negative_slope_indices.extend(segment_indices)
            negative_slopes.extend(dvdtslope[segment_indices - 1])

    negative_slope_indices = np.array(negative_slope_indices)
    negative_slopes = np.array(negative_slopes)

    negative_slope_indices, unique_indices = np.unique(negative_slope_indices, return_index=True)
    negative_slopes = np.array(negative_slopes)[unique_indices]

   
    change_in_slopes = np.diff(negative_slopes)
    negative_slopes_truncated=negative_slopes[:-1]

    change_in_slopes=change_in_slopes[1:]
    negative_slope_indices = negative_slope_indices[1:] 
    negative_slopes_truncated=negative_slopes_truncated[1:]
    
    least_change_index = np.argmin(np.abs(change_in_slopes) + np.abs(negative_slopes_truncated)) 
    dvdt_at_least_change = filtered_dvdt[negative_slope_indices[least_change_index]] 
    
    negative_slope_indices = negative_slope_indices[1:] 
  
    fig, ax1 = plt.subplots()
    ax1.plot(filtered_dvdt, label='Filtered DVDT')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('DVDT')
    ax1.axvline(x=negative_slope_indices[least_change_index], color='r', linestyle='--', label='Least Change in Slope')
    ax1.scatter(negative_slope_indices[least_change_index], filtered_volt_segment[negative_slope_indices[least_change_index]], color='r', label='Least Change Point')
    ax1.legend(loc='upper right')

    ax1.annotate(f'dvdt: {dvdt_at_least_change:.2f}', 
                xy=(negative_slope_indices[least_change_index], filtered_volt_segment[negative_slope_indices[least_change_index]]),
                xytext=(negative_slope_indices[least_change_index] + 5, filtered_volt_segment[negative_slope_indices[least_change_index]] + 5))
    ax2 = ax1.twinx()
    ax2.plot(negative_slope_indices, negative_slopes_truncated, label='Negative Slopes', color='g',linewidth=0.2)

    ax2.set_ylabel('Negative Slopes')
    ax2.legend(loc='lower left')

    plt.title('Filtered Voltage Segment with Least Change in Slope')

    # Save the plot as a PDF
    fig.savefig(f'{mut_name}_dvdt_slopes.pdf')
    #### End shoulder-finding code ####

    curr_peaks_indices,curr_peaks_values= find_peaks(dvdt,height = 100)
    print(f'start: {start}, start2:{start2}, end: {end}')
    print(f'volt_segment: {volt_segment}')
    print(f'dvdt: {dvdt}')
    print(f'curr_peaks_indices: {curr_peaks_indices}')
    print(f'curr_peaks_values: {curr_peaks_values}')
    features[0]['dvdt Peak1 Height'] = curr_peaks_values['peak_heights'][0]
    features[0]['dvdt Peak1 Voltage'] = volt_segment[curr_peaks_indices[0]] 
    features[0]['dvdt Peak2 Height'] = curr_peaks_values['peak_heights'][-1]
    features[0]['dvdt Peak2 Voltage'] = volt_segment[curr_peaks_indices[-1]]
    features[0]['dvdt Threshold'] = volt_segment[np.where(dvdt>1)[0][0]]
    features[0]['dvdt Peak2 Shoulder'] = dvdt_at_least_change
    # features[0]['Peak2_shoulder'] = peak2

    positive_slope_indices = np.where(dvdt > 1)[0]
    if len(positive_slope_indices) > 0:
        threshold_index = positive_slope_indices[0]
        features[0]['dvdt Threshold_DEBUG'] = volt_segment[threshold_index]
    else:
        features[0]['dvdt Threshold_DEBUG'] = None
    
    if rec_extra:
    #for ais
        trace['V'] = extra_vms['ais']
        feat_list = ['Spikecount']
        traces = [trace]
        feature_ais = efel.getFeatureValues(traces,feat_list)
        features[0]['ais spikecount'] =feature_ais[0]['Spikecount']
        features[0]['ais spikecount'] =features[0]['ais spikecount'][0]
        
        #for nexus
        trace['V'] = extra_vms['nexus']
        feat_list = ['Spikecount']
        traces = [trace]
        feature_nex = efel.getFeatureValues(traces,feat_list)
        features[0]['nex spikecount'] =feature_nex[0]['Spikecount']
        features[0]['nex spikecount'] =features[0]['nex spikecount'][0]
        
        #for dist_dend
        trace['V'] = extra_vms['dist_dend']
        feat_list = ['Spikecount']
        traces = [trace]
        feature_disdend = efel.getFeatureValues(traces,feat_list)
        features[0]['disdend spikecount'] =feature_disdend[0]['Spikecount']
        features[0]['disdend spikecount'] =features[0]['disdend spikecount'][0]
    
    features = pd.DataFrame(features)
    features = features.drop(columns =['all_ISI_values'])
    features.insert(0,'Type',mut_name)
    with open (f'{prefix}_efel.csv','w') as f:
        features.to_csv(f,index=False)
    f.close()
    return features
   

mut_names = ['']
mut_not_found = {}
feature_row=None
for mut_name in mut_names:
    try:
        feature_row = get_features(mutant_name=mut_name)
        with open('efel_features.csv', 'a') as f:
            feature_row.to_csv(f, header=f.tell()==0,index=False) #bug needs an existing file

    except Exception as e:
        print(e)
        mut_not_found[mut_name] = e



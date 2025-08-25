from Na12HMMModel_TF import *
import matplotlib.pyplot as plt
import numpy as np
import NrnHelper as NH
from neuron import h
import time
import efel
efel.api.setDoubleSetting('Threshold', -40) #15 originally. ##TF022625 changed to -5 for kevin response to reviewers
import pandas as pd
import math
from scipy.signal import find_peaks

def get_sim_volt_values(sim,mut_name,rec_extra = False,dt = 0.005,stim_amp = 0.5): #originally had mutant_name, changed to mut_name 121223TF #dt=0.005 Original stim_amp=0.5 original

    # sim = Na12Model_TF(mutant_name)
    #sim = Na12Model_TF(mut_name)
    sim.dt= dt
    #sim.make_het()
    rec_extra = True
    sim.l5mdl.init_stim(amp=stim_amp)
    if rec_extra:
        Vm, I, t, stim,extra_vms = sim.l5mdl.run_model(dt=dt,rec_extra = rec_extra)

        sim.extra_vms = extra_vms
    else:
        Vm, I, t, stim = sim.l5mdl.run_model(dt=dt)

        extra_vms = {}

    return Vm,t,extra_vms,I,stim


def get_sim_volt_valuesTF(sim,mut_name,dt = 0.005,stim_amp = 0.5): #originally had mutant_name, changed to mut_name 121223TF #dt=0.005 Original

    # sim = Na12Model_TF(mutant_name)
    #sim = Na12Model_TF(mut_name)
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


def get_features(sim,prefix=None,mut_name = 'na12annaTFHH2',rec_extra=True): #added sim_config to allow run_sim_model instead of run_model 011924TF
    print("running routine")
    dt=0.005#0.1#0.005
    Vm,t,extra_vms,_,__ = get_sim_volt_values(sim,mut_name,rec_extra=rec_extra)
    # Vm,t,I,_ = get_sim_volt_valuesTF(sim,mut_name)
    #creating the trace file
    stim_start = 200 #100 original
    stim_end = 1900 #800 original
    trace={}
    trace = {'T':t,'V':Vm,'stim_start':[stim_start],'stim_end':[stim_end]}
    trace['T']= trace['T'] * 1000
    #for neu
    # feature_list= ['AP_height','AP_width','AP1_peak','AP1_width','Spikecount','all_ISI_values'] #original
    feature_list= ['AP_height','AP_width','AP1_peak','AP1_width','Spikecount','all_ISI_values'] ## 022525 abbreviated features for kevin response to reviewers
    
    traces = [trace]
    features = efel.getFeatureValues(traces,feature_list)
    
    ###Plotting Voltages to debug not getting enough spikes ##TF111524
    # plt.plot(trace['T'], trace['V'])
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Voltage (mV)')
    # plt.title('Voltage vs Time')
    # plt.savefig('ZZZZZZZZZZZZZ_OGrun_voltage_vs_time5.pdf', format='pdf')  # Replace with your desired filename
    # plt.close()
    ###Plotting Voltages to debug not getting enough spikes ##TF111524

    try:
        features[0]['ISI mean'] =features[0]['all_ISI_values'].mean()
    except Exception as e:
        features[0]['ISI mean'] = 0
    features[0]['AP_height'] =features[0]['AP_height'].mean()
    features[0]['AP_width'] =features[0]['AP_width'][0].mean()
    features[0]['AP1_peak'] =features[0]['AP1_peak'][0]
    features[0]['AP1_width'] =features[0]['AP1_width'][0]
    features[0]['Spikecount'] =features[0]['Spikecount'][0]
    #import pdb; pdb.set_trace()
    #for dv/dt Peak 1 and Peak 2 and sum
    spike_count= features[0]['Spikecount']
    isi_values = features[0]['all_ISI_values']
    median_spike = int(math.floor(spike_count/2)) + 1
    #median spike location
    print(f'Length of isi_values {len(isi_values)}')
    print(f'isi_values: {isi_values}')
    print(f'Spike Count = {spike_count}')

    
    
    ## efel starting points. Take the stimulus start and add the sum of dictated isi values to get start. Then divide by dt to get time step
    start = int((stim_start + isi_values[0:median_spike-1].sum())/dt) #original    #dividing by dt to get into same unit
    start2 = int((stim_start + isi_values[0:5].sum())/dt) ## start about 6 spikes in
    start3 = int((stim_start+isi_values[1])/dt) ## start about 1 spike in.
    start4 = int((stim_start+isi_values[0:3].sum())/dt) ## start about 3 spikes in.
    try:
        # end = start + int(isi_values[median_spike]/dt)
        end = 380000
    except Exception as e:
        end=400000
        print("There were not enough spikes to calculate median isi")

    volt_segment = Vm[start:end] ## get voltage values for dictated segment
    
    dvdt = np.gradient(volt_segment) / dt ## calculate dvdt






    ###----------------------------------------###
    ### TF052225 FIND SHOULDER BEFORE PEAK 2 ###
    ###----------------------------------------###
    curr_peaks_indices,curr_peaks_values= find_peaks(dvdt,height = 100) ##original
    # curr_peaks_indices,curr_peaks_values= find_peaks(dvdt,height = 10) ##TF022625 reducing height for kevin response to reviewers
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
    
    #### TF052125 Find shoulder BEFORE dvdt peak 2 ####
    peak2_index = curr_peaks_indices[-1]  # Index of dvdt Peak 2
    
    # Look for the minimum of the second derivative BEFORE peak 2
    # Define a window before peak 2 to search for the shoulder
    window_before_peak2 = 20 # Adjust as needed
    
    # Ensure the window doesn't go out of bounds
    start_index = max(0, peak2_index - window_before_peak2)
    
    # Calculate the second derivative of dvdt
    dvdt2 = np.gradient(dvdt) / dt
    
    # Find the index of the minimum second derivative within the window
    shoulder_index_before_peak2 = start_index + np.argmin(dvdt2[start_index:peak2_index])
    
    # Get the dvdt value at the shoulder
    dvdt_at_shoulder_before_peak2 = dvdt[shoulder_index_before_peak2]
    
    features[0]['dvdt Peak2 Shoulder Before'] = dvdt_at_shoulder_before_peak2
    features[0]['dvdt Peak2 Shoulder Before Voltage'] = volt_segment[shoulder_index_before_peak2]
    
    print(f'Shoulder Index Before Peak 2: {shoulder_index_before_peak2}')
    print(f'dvdt value at shoulder before Peak 2: {dvdt_at_shoulder_before_peak2}')
    ###----------------------------------------###






    ##-----------------------------------------------------------------------------------------------------##
    #### TF030425 Find shoulders when dvdt peak 2 is not a true peak (adjacent values with local maxima) AFTER PEAK2 ####
    ##-----------------------------------------------------------------------------------------------------##
    filtered_indices = np.where(dvdt > 50)[0] # Filter dvdt values greater than threshold (50) 
    filtered_dvdt = dvdt[filtered_indices] # get dvdt values at the filtered indices
    filtered_volt_segment = volt_segment[filtered_indices] # get voltage values at the filtered indices

    dvdtslope = np.diff(filtered_dvdt) # Calculate the slope of filtered dvdt values

    peaks, peaks_vals = find_peaks(filtered_dvdt) # Identify peaks in filtered_dvdt

    negative_slope_indices = []
    negative_slopes = []

    # Iterate through peaks to find negative slope segments (start 5 time points to the right of each peak)
    n=5
    for peak in peaks:
        start_index = peak + n # Start looking for negative slopes n points to the right of each peak
        if start_index < len(dvdtslope):            
            segment_indices = np.where(dvdtslope[start_index:] < 0)[0] + start_index + 1 # Find the segment where the slope is negative after the start_index
            segment_indices = segment_indices[segment_indices < len(filtered_dvdt)]
            negative_slope_indices.extend(segment_indices)
            negative_slopes.extend(dvdtslope[segment_indices - 1])

    # Convert lists to numpy arrays
    negative_slope_indices = np.array(negative_slope_indices)
    negative_slopes = np.array(negative_slopes)

    # negative_slope_indices = np.array(negative_slope_indices)
    negative_slope_indices, unique_indices = np.unique(negative_slope_indices, return_index=True)
    negative_slopes = np.array(negative_slopes)[unique_indices]

    # calculate the change in slopes (second derivative)
    change_in_slopes = np.diff(negative_slopes)
    negative_slopes_truncated=negative_slopes[:-1]

    change_in_slopes=change_in_slopes[1:]
    negative_slope_indices = negative_slope_indices[1:] ## exclude first value since it could be slope closer to 0
    negative_slopes_truncated=negative_slopes_truncated[1:]
    
    least_change_index = np.argmin(np.abs(change_in_slopes) + np.abs(negative_slopes_truncated)) # look for index where change in slope is closest to zero AND slope is negative
    dvdt_at_least_change = filtered_dvdt[negative_slope_indices[least_change_index]] # Get the dvdt value at the point where the change in slope is closest to zero
    
    negative_slope_indices = negative_slope_indices[1:] ## exclude first value since it could be slope closer to 0

   # Print results
    print(f'Filtered peaks: {peaks}, Filtered peak values: {peaks_vals}')
    print(f'Filtered dvdt values: {filtered_dvdt}')
    print(f'Index of least change in slope: {least_change_index}')
    print(f'dvdt value at least change in slope: {dvdt_at_least_change}')
    print(f'length of negative_slope_indices: {len(negative_slope_indices)}')
    print(f'length of negative_slopes: {len(negative_slopes)}')
    print(f'length of change_in_slopes: {len(change_in_slopes)}')
    print(f'length of negative_slopes_truncated: {len(negative_slopes_truncated)}')

    ## Plotting of filtered dvdt vs slope (1st vs 2nd derivative) to see point at which slope is lowester to identify peak 2 shoulder
    # Plot filtered_dvdt on the primary y-axis
    fig, ax1 = plt.subplots()
    ax1.plot(filtered_dvdt, label='Filtered DVDT')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('DVDT')
    ax1.axvline(x=negative_slope_indices[least_change_index], color='r', linestyle='--', label='Shoulder After Peak 2')
    
    # Find the index of shoulder_index_before_peak2 in filtered_indices
    try:
        filtered_shoulder_index = np.where(filtered_indices == shoulder_index_before_peak2)[0][0]
        ax1.axvline(x=filtered_shoulder_index, color='violet', linestyle=':', label='Shoulder Before Peak 2')
        ax1.annotate(f'peak1 shoulder: {dvdt_at_shoulder_before_peak2:.2f}',
                    xy=(filtered_shoulder_index, filtered_dvdt[filtered_shoulder_index]),
                    xytext=(filtered_shoulder_index + 5, filtered_dvdt[filtered_shoulder_index] + 5), fontsize=6)
    except IndexError:
        print("Shoulder index not found in filtered_indices")    
    ax1.scatter(negative_slope_indices[least_change_index], filtered_volt_segment[negative_slope_indices[least_change_index]], color='r', label='Shoulder After Peak 2')
    ax1.legend(loc='upper left', fontsize=6)

    # Annotate the dvdt value at the point where the change in slope is least
    ax1.annotate(f'dvdt: {dvdt_at_least_change:.2f}', 
                xy=(negative_slope_indices[least_change_index], filtered_volt_segment[negative_slope_indices[least_change_index]]),
                xytext=(negative_slope_indices[least_change_index] + 5, filtered_volt_segment[negative_slope_indices[least_change_index]] + 5), fontsize=6)
    # Create a second y-axis for negative_slopes
    ax2 = ax1.twinx()
    # ax2.plot(negative_slopes, label='Negative Slopes', color='g', linewidth=0.2)
    ax2.plot(negative_slope_indices, negative_slopes_truncated, label='Negative Slopes', color='g',linewidth=0.2)

    ax2.set_ylabel('Negative Slopes')
    ax2.legend(loc='lower left', fontsize=6)

    plt.title('Filtered Voltage Segment with Least Change in Slope')

    # Save the plot as a PDF
    fig.savefig(f'{prefix}{mut_name}_dvdt_slopes.pdf')
    ##-----------------------------------------------------------------------------------------------------##
    #### End shoulder-finding code ####
    ##-----------------------------------------------------------------------------------------------------##



    curr_peaks_indices,curr_peaks_values= find_peaks(dvdt,height = 100) ##original
    # curr_peaks_indices,curr_peaks_values= find_peaks(dvdt,height = 10) ##TF022625 reducing height for kevin response to reviewers
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
    features[0]['dvdt Peak2 Shoulder After'] = dvdt_at_least_change
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
    #features.to_csv(f'{paramlog_folder_path}/combined_paramlogs_{out_sfx}.csv', index=False) #Modify for saving features to csv
    return features
    




# mut_names = ['R853Q','E1211K','A1773T','G879R','A880S','A427D','E430A','E999K','E1211K','E1880K','G879R',
#            'K1260E','K1260Q','M1879T','R571H','R850P','R1319L','R1626Q','R1882L','R1882Q','S1780I','Y816F','na12WT2']

mut_names = ['na12_HMM_TF100923']
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



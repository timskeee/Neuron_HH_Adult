import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import your custom modules
import Na12HMMModel_TF as tf
from NrnHelper import *
from vm_plotter import *
from scalebary import *
import fitz  # PyMuPDF for PDF handling

#############################################################
# Function to combine all PDF files in a folder into one multi-page pdf
#############################################################
def combine_pdfs_with_header(folder_path, output_filename="combined.pdf", output_folder=None):
  """
  Combine all PDFs in folder_path into a single PDF with headers.
  Save the result in output_folder (if provided), else in folder_path.
  """
  pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
  pdf_files.sort()
  output = fitz.open()
  for pdf in pdf_files:
    full_path = os.path.join(folder_path, pdf)
    src = fitz.open(full_path)
    for page in src:
      # Draw filename at the top of each page
      page.insert_text((10, 10), pdf, fontsize=4, color=(0, 0, 0))
      output.insert_pdf(src, from_page=page.number, to_page=page.number)
    src.close()
  if output_folder is None:
    output_path = os.path.join(folder_path, output_filename)
  else:
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    output_path = os.path.join(output_folder, output_filename)
  output.save(output_path)
  output.close()
  print(f"Combined {len(pdf_files)} PDFs into {output_path}")
#############################################################
#############################################################

# Set up paths
root_path_out = '/global/homes/t/tfenton/Neuron_general-2/Plots//12HH16HH/2-DevelopingBranch/6-paramsearch'
path = 'developing_neuron_tuning'

# Create output directories
out_path1 = os.path.join(root_path_out, path, 'dvdt')
out_path2 = os.path.join(root_path_out, path, 'currentscapes')
out_path3 = os.path.join(root_path_out, path, 'input_resistance')
tuning_path = os.path.join(root_path_out, path, 'tuning')

# for p in [out_path1, out_path2, out_path3, tuning_path]:
for p in [tuning_path]:
    if not os.path.exists(p):
        os.makedirs(p)

def cm_to_in(cm):
    return cm/2.54

def tune_parameters_for_input_resistance(target_min=800, target_max=1200, max_iterations=1500):
    """
    Automatically tune parameters to achieve target input resistance.
    
    Parameters:
    target_min, target_max: Target input resistance range in MOhms
    max_iterations: Maximum number of parameter adjustments to try
    
    Returns:
    best_params: Dictionary of best parameter values found
    best_resistance: The input resistance achieved with best_params
    """
    
    # Parameter ranges with MORE AGGRESSIVE changes to see actual differences
    param_ranges = {
        'fac': [0.05],                    # Wider gpas range 
        'fac2': [0.005, 0.01, 0.02],                  # Wider Ih range
        'f2': [0.001,0.01,0.1,0.2,0.3],                   # Much larger K+ range to see real effects
        'nav12factor': [0.3,0.35,0.4],        # Lower Na+ range for significant amplitude reduction
        'aisnav': [],
        'somanav': []
    }
    
    best_params = {'fac': 0.5, 'fac2': 1, 'f2': 0.1, 'nav12factor': 0.5}
    best_resistance = float('inf')
    best_distance = float('inf')
    
    print(f"Starting parameter tuning for input resistance target: {target_min}-{target_max} MΩ")
    print(f"Will test up to {max_iterations} parameter combinations")
    
    iteration = 20
    results_log = []
    
    # Grid search through parameter combinations
    for fac in param_ranges['fac']:
        for fac2 in param_ranges['fac2']:
            for f2 in param_ranges['f2']:
                for nav12factor in param_ranges['nav12factor']:
                    if iteration >= max_iterations:
                        print(f"Reached maximum iterations ({max_iterations}), stopping search")
                        break
                    
                    iteration += 1
                    print(f"\nIteration {iteration}/{max_iterations}: Testing fac={fac}, fac2={fac2}, f2={f2}, nav12factor={nav12factor}")
                    
                    try:
                        # Create model with MORE AGGRESSIVE parameter changes
                        simwt = tf.Na12Model_TF(
                            ais_nav12_fac=nav12factor*3,        # Reduced from 6 to 3 for much lower AIS excitability
                            nav12=nav12factor,
                            ais_nav16_fac=nav12factor*3,        # Reduced from 6 to 3 for much lower AIS excitability
                            nav16=nav12factor,
                            soma_na16=nav12factor*1.0,          # Reduced from 2.5 to 1.0 for much lower somatic excitability
                            soma_na12=nav12factor*1.0,          # Reduced from 2.5 to 1.0 for much lower somatic excitability
                            node_na=0.8*nav12factor,            # Reduced from 1.5 to 0.8 for lower node excitability
                            dend_nav12=nav12factor*0.5,         # Reduced from 1.2 to 0.5 for minimal dendritic excitability
                            somaK=5.0*f2,                       # Increased from 1.8 to 5.0 for much stronger repolarization
                            K=3.0*f2,                           # Increased from 1.5 to 3.0 for stronger general K+
                            KP=3.0*f2,                          # Increased from 1.3 to 3.0 for much stronger delayed rectifier
                            KT=2.5*f2,                          # Increased from 1.2 to 2.5 for stronger transient K+
                            ais_ca=0.3*f2,                      # Reduced from 0.6 to 0.3 to minimize Ca2+ buildup
                            ais_Kca=2.0*f2,                     # Increased from 1.0 to 2.0 for strong afterhyperpolarization control
                            na12name='na12annaTFHH2',
                            mut_name='na12annaTFHH2',
                            na12mechs=['na12','na12mut'],
                            na16name='na12annaTFHH2',
                            na16mut_name='na12annaTFHH2',
                            na16mechs=['na16','na16mut'],
                            params_folder='./params/',
                            plots_folder=f'{root_path_out}/{path}',
                            update=True,
                            fac=fac,
                            fac2=fac2
                        )
                        
                        # Test input resistance with hyperpolarizing currents for linear measurement
                        # Also use longer duration to see steady-state after Ih activation
                        _, _, voltages_at_time, resistances = plot_input_resistance(
                            simwt,
                            stim_amps=[-0.1,-0.09,-0.08,-0.07, -0.06,-0.05, -0.04, -0.03, -0.02, -0.01,
                                       0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],  
                            plot_fn=f'tuning/{iteration}_gpas_{fac}_cm_{fac2}_K_{f2}_nav12factor_{nav12factor}_Ra150Ih8e-12',
                            dt=0.1,
                            stim_dur=600,  # Longer to reach steady state with Ih
                            v_time=550     # Measure near end for true steady state
                        )
                        # simwt.make_currentscape_plot(amp=0.05, time1=0,time2=250,stim_start=100, sweep_len=250, pfx=f'currentscapes/{namestr}_')

                        # DEBUG: Check units and values
                        print(f"  DEBUG - Voltage range: {min(voltages_at_time):.3f} to {max(voltages_at_time):.3f}")
                        print(f"  DEBUG - Raw resistances: {[f'{r:.6f}' for r in resistances[:3]]}")
                        print(f"  DEBUG - Current range: {min([-0.05, -0.04, -0.03, -0.02, -0.01,0.01,0.02,0.03,0.04,0.05]):.3f} to {max([-0.05, -0.04, -0.03, -0.02, -0.01,0.01,0.02,0.03,0.04,0.05]):.3f} nA")
                        
                        # Calculate average resistance magnitude (excluding NaN values)
                        valid_resistances = [abs(r) for r in resistances if not np.isnan(r) and r != 0]
                        if valid_resistances:
                            avg_resistance = np.mean(valid_resistances)
                            
                            # Calculate distance from target range
                            if target_min <= avg_resistance <= target_max:
                                distance = 0  # Perfect hit
                            elif avg_resistance < target_min:
                                distance = target_min - avg_resistance
                            else:
                                distance = avg_resistance - target_max
                            
                            print(f"  Input resistance: {avg_resistance:.1f} MΩ, Distance from target: {distance:.1f}")
                            
                            # Log results
                            results_log.append({
                                'iteration': iteration,
                                'fac': fac,
                                'fac2': fac2,
                                'f2': f2,
                                'nav12factor': nav12factor,
                                'resistance': avg_resistance,
                                'distance': distance
                            })
                            
                            # Update best parameters if this is better
                            if distance < best_distance:
                                best_distance = distance
                                best_resistance = avg_resistance
                                best_params = {
                                    'fac': fac,
                                    'fac2': fac2,
                                    'f2': f2,
                                    'nav12factor': nav12factor
                                }
                                print(f"  *** NEW BEST! *** Resistance: {best_resistance:.1f} MΩ, Distance: {best_distance:.1f}")
                                
                                # Continue searching even if we hit target to find potentially better solutions
                        else:
                            print(f"  No valid resistance measurements obtained")
                        
                        # Clean up to avoid memory issues
                        del simwt
                        
                    except Exception as e:
                        print(f"  ❌ Error with parameters: {e}")
                        continue
    
    # Save results log
    results_df = pd.DataFrame(results_log)
    results_df.to_csv(f'{tuning_path}/parameter_tuning_log.csv', index=False)
    combine_pdfs_with_header(folder_path=tuning_path, output_filename=f"Rin_combined.pdf", output_folder=f"{root_path_out}/{path}/tuning")

    input('Press Enter to continue after reviewing the tuning log...')


    print(f"\n{'='*60}")
    print(f"PARAMETER TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best parameters found:")
    print(f"  fac (gpas): {best_params['fac']}")
    print(f"  fac2 (Ih): {best_params['fac2']}")
    print(f"  f2 (K+ conductance): {best_params['f2']}")
    print(f"  nav12factor (Na+ conductance): {best_params['nav12factor']}")
    print(f"  Achieved input resistance: {best_resistance:.1f} MΩ")
    print(f"  Distance from target range: {best_distance:.1f} MΩ")
    print(f"  Total iterations tested: {iteration}")
    
    return best_params, best_resistance

def main():
    """Main execution function - simplified for tuning only"""
    
    print("🧠 NEURON PARAMETER TUNING FOR DEVELOPING MODEL")
    print("=" * 60)
    print("Goal: Find parameters for sustained firing with proper repolarization")
    print()
    
    # Run parameter tuning only
    print("🔍 AUTOMATED PARAMETER TUNING")
    best_params, best_resistance = tune_parameters_for_input_resistance(
        target_min=800, 
        target_max=1200, 
        max_iterations=1500
    )
    
    return best_params, best_resistance

# Run the optimization if this script is executed directly
if __name__ == "__main__":
    try:
        best_params, best_resistance = main()
        print("\n🎊 Parameter tuning completed successfully!")
    except Exception as e:
        print(f"\n❌ Script failed with error: {e}")
        import traceback
        traceback.print_exc()


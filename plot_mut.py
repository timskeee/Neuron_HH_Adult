from NaMut import *
import NrnHelper as nhlpr
import os 
        
ic_fitter_path = '/global/homes/r/roybens/IC_Fitter/'
data_folder = '/global/cfs/cdirs/m2043/roybens/na_pkl_v1/'
ic_fitter_script = 'plot_from_pkl.py'
cwd = os.getcwd()  
def plot_mut(mut_name):
    #initialize WT and stuff
    dt = 0.005
    amp = 0.3
    st_fi = 0
    end_fi = 0.5
    n_fi = 11
    fi_xaxis = np.linspace(st_fi,end_fi,n_fi)
    wt_fis = []
    
    sim = NaMut(mut_name,params_folder = f'{data_folder}/Neuron_general/params/',plots_folder = f'{data_folder}/plots/' )
    sim.make_wt()

    #plot volts and dvdt same figure
    fig_volts,axs_volts = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
    sim.plot_stim(axs = axs_volts[0],dt=dt,stim_amp = amp,rec_extra = True)
    #updating the sim (NaMut) to update hold the simulation so we won't need to redo it.
    sim.volt_soma_wt = sim.volt_soma 
    sim.I_wt = sim.I
    sim.extra_vms_wt = sim.extra_vms
    nhlpr.plot_dvdt_from_volts(sim.volt_soma_wt,dt = dt,skip_first = True,axs=axs_volts[1])
    fig_extra_vs,extra_vs_axs = plt.subplots(3,figsize=(cm_to_in(8),cm_to_in(23)))
    plot_extra_volts(sim.t,sim.extra_vms_wt,axs =extra_vs_axs)

    wt_fis = nhlpr.get_fi_curve(sim.l5mdl,st_fi,end_fi,n_fi)


    sim.make_het()
    sim.plot_stim(axs = axs_volts[0],dt=dt,stim_amp = amp,rec_extra = True,clr = 'red')
    nhlpr.plot_dvdt_from_volts(sim.volt_soma,dt = dt,skip_first = True,axs=axs_volts[1],clr = 'red')
    plot_extra_volts(sim.t,sim.extra_vms,axs =extra_vs_axs ,clr = 'red')

    fig_volts.savefig(f'{sim.plot_folder}{mut_name}_Vs_dvdt.pdf')
    fig_extra_vs.savefig(f'{sim.plot_folder}{mut_name}_ext_Vs.pdf')

    nhlpr.get_fi_curve(sim.l5mdl,st_fi,end_fi,n_fi,wt_data = wt_fis,fn = f'{sim.plot_folder}{mut_name}_FI.pdf')

def analyze_mut(mut_name):
    os.chdir(ic_fitter_path)
    os.system(f'python3 {ic_fitter_script} {mut_name}')
    os.chdir(cwd)
    plot_mut(mut_name)
def analyze_all():
    not_good = []
    mut_lists = ['S1780I','M1879T','E1880K','R1882L','R1882Q','R853Q','D1050V','E1211K','K1422E','S1758R','A1773T','A427D','E430A','R571H','Y816F','G879R','A880S','G882E','F978L','D997G','E999K','K1260E','K1260Q','R1319L','Q1479P','R1626Q']
    for curr_mut in mut_lists:
        try:
            analyze_mut(curr_mut)
        except Exception as e:
            print(f'could not produce {curr_mut}')
            print(e)
            not_good.append(curr_mut)
    print('***** did not run -'+not_good)
analyze_all()


#plot_mut('A1773T')
#analyze_mut('Y816F')
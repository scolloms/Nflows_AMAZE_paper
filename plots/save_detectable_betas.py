from plot_functions import save_detectable_betas
import glob

files = extra_files = glob.glob('/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/cont_GWTC3/prod_retrainedCE/*.hdf5')
extra_files = glob.glob('/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/cont_GWTC3/prod_longsampling/*.hdf5')
analysis_name = 'continuous_combined'
outdir = '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/plots/prod_091224/'
save_detectable_betas(files+extra_files, analysis_name, outdir)
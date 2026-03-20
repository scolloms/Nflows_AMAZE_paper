tar -cvf plot_data.tar --transform='s,plots/prod_091224/data/dataspace_samps_extrafiles,plot_data/dataspace_samps,' ../plots/prod_091224/data/dataspace_samps_extrafiles.hdf5
tar -rvf plot_data.tar --transform='s,outputs,plot_data,' ../outputs/KLs_KSs.json
tar -rvf plot_data.tar paper_plots_flow.ipynb
tar -rvf plot_data.tar mpl.sty
tar -rvf plot_data.tar zoom_plot.py
gzip plot_data.tar
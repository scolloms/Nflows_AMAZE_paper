tar -cvf plot_data.tar --transform='s,plots/prod_091224/data,plot_data,' ../plots/prod_091224/data/dataspace_samps.hdf5
tar -rvf plot_data.tar --transform='s,plots/prod_091224/data,plot_data,' ../plots/prod_091224/data/emulation_samps.hdf5
tar -rvf plot_data.tar --transform='s,plots/prod_091224/data,plot_data,' ../plots/prod_091224/data/test_flow_samps.hdf5
tar -rvf plot_data.tar --transform='s,plots/prod_091224/data,plot_data,' ../plots/prod_091224/data/KLs.json
gzip plot_data.tar
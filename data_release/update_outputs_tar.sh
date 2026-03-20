

gunzip inference_results.tar.gz
tar -rvf inference_results.tar --transform='s,outputs/cont_GWTC3/prod_longsampling,cont_GWTC3,' ../outputs/cont_GWTC3/prod_longsampling/*.hdf5
tar -vf inference_results.tar --delete cont_detectable_samps/cont_detectable.hdf5
tar -rvf inference_results.tar --transform='s,plots/prod_091224/data/cont_combined_detectable_betas.hdf5,cont_detectable_samps/cont_detectable.hdf5,' ../plots/prod_091224/data/cont_combined_detectable_betas.hdf5
gzip inference_results.tar
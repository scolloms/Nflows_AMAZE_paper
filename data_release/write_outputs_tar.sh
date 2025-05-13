

tar -cvf inference_results.tar --transform='s,outputs/cont_GWTC3/prod_retrainedCE,cont_GWTC3,' ../outputs/cont_GWTC3/prod_retrainedCE/*hdf5
tar -rvf inference_results.tar --transform='s,outputs/discrete_GWTC3/flow_retrainedCE,discrete_GWTC3/flow,' ../outputs/discrete_GWTC3/flow_retrainedCE/*hdf5
tar -rvf inference_results.tar --transform='s,outputs/discrete_GWTC3/KDEs,discrete_GWTC3/KDEs,' ../outputs/discrete_GWTC3/KDEs/*hdf5
tar -rvf inference_results.tar --transform='s,plots/prod_091224/data/cont_retrainedCE_detectable_betas.hdf5,cont_detectable_samps/cont_detectable.hdf5,' ../plots/prod_091224/data/cont_retrainedCE_detectable_betas.hdf5
gzip inference_results.tar
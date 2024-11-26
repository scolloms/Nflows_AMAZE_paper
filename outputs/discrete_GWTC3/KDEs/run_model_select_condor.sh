#! /bin/bash
#runs model select with cpu, with flows, and preset file paths, CE channel training
#running inference with different event posterior seed, different sampler seed, and 500 walkers each with 1000 iterations

seed = $1
echo $1
model_path="/data/wiay/2297403c/models_reduced.hdf5"
gw_path="/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/inputs/GWTC-3/events"

/data/wiay/2297403c/conda_envs/amaze/bin/python /data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/model_select --file-path ${model_path} \
        --model0 'gwobs' \
        --gw-path ${gw_path} \
	--verbose \
        --channels 'CE' 'CHE' 'GC' 'NSC' 'SMT'\
        --sensitivity 'midhighlatelow_network' \
        --save-samples \
        --prior 'p_theta_jcb' \
        --regularisation_N '990903' \
        --name seed$1 \
        --Nsamps 11058 \
        --random-seed $1 \
        --multiproc
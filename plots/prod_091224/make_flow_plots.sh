#!/bin/bash
flow_path="/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/inputs/flow_models/mixed_models/"

/data/wiay/2297403c/conda_envs/amaze/bin/python /data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/plot/flow_results/make_paper_plots.py \
    --flow-path ${flow_path} \
    --channel-label 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
    --hyperparam-idxs 0 2 \
    --conditional 0.0 1.0 \
    --plot-discrete-result \
    --justplot \
    --KDE-result-path '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/discrete_GWTC3/KDEs/' \
    --discrete-result-path '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/discrete_GWTC3/flow_retrainedCE/' \
    --cont-result-path '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/cont_GWTC3/prod_retrainedCE/' \
    --outdir '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/plots/prod_091224' \
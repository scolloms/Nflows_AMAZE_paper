#!/bin/bash
flow_path="/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/inputs/flow_models/mixed_models/"

/data/wiay/2297403c/conda_envs/amaze/bin/python /data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/plots/make_paper_plots.py \
    --flow-path ${flow_path} \
    --plot-cont-result \
    --justplot \
    --KDE-result-path '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/discrete_GWTC3/KDEs/' \
    --discrete-result-path '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/discrete_GWTC3/flow_retrainedCE/' \
    --cont-result-path '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/cont_GWTC3/prod_retrainedCE/' \
    --cont-result-path-extra '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/cont_GWTC3/prod_longsampling/' \
    --outdir '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/plots/prod_091224' \
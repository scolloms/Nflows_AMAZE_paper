flow_path="/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/inputs/flow_models/CE_test_population/"

python /data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/plot/flow_results/make_paper_plots.py \
    --flow-path ${flow_path} \
    --channel-label 'CE' \
    --hyperparam-idxs 1 2 \
    --conditional 0.1 1.0 \
    --plot-flow-corner \
    --justplot \
    --outdir '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/plots/prod_091224' \
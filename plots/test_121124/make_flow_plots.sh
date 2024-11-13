flow_path="/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/inputs/flow_models/test_121124/"

python /data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/plot/flow_results/make_paper_plots.py \
    --flow-path ${flow_path} \
    --channel-label 'CE' 'CHE' 'GC' 'NSC' 'SMT' \
    --hyperparam-idxs 1 4 \
    --conditional 0.1 5.0 \
    --outdir '/data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/plots/test_121124' \
    --justplot
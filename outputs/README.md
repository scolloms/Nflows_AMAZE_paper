18-11-24

Checking in on my tests (250 walkers and 1000 steps per walker), the continuous flows run is at 90% done after running for 6 days. Unfortunately I haven't saved the burnin for these, so will check each chain after and check if they have converged already or need to increase the burnin fraction.

The test KDE run is going to take forver, it's freezing probabilities for model 5/36 after 3 days. This will not depend on the number of walkers so it's just number of posterior samples that needs reduced. My optimisation in /notebooks/plot_obs_likelihood.ipynb suggested 11058 samples, so I will delete the current run and try that.

Otherwise setting up a run to train the CE model with a test population model removed. The outputs from this will live in /CE_test_population, and flow models in /inputs/CE_test_population. Running this with bf0bdd13f746facbdd6bc04737342312fbd3bd78 code ver.

19-11-24

Checked burnin for test run at /data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/cont_GWTC3/test_121124/plot_chains.ipynb
There is some burnin still present here, I think I will increase the burnin fraction from 0.2 (the test run) to 0.4 for the production runs.

The CE test smdl run also finished and I plotted the trained flow on the test model, it looks good.

I think I will use the trained flows I have already for inference, just copy over the test_121124 models, as I haven't made any changes to the training since running these tests. Running production runs with code ver 2955e7db64ed4bcd977fa424affd17c2c8012093 in cont_GWTC3/production.

And now in discrete_GWTC3/flow and /KDEs.

22-11-24

First continuous inference run done and looked at results in /data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/notebooks/plot_continuous_result.ipynb. Trying to figure out why the alpha_CE posterior is quite different. The last results I had in /data/wiay/2297403c/amaze_model_select/AMAZE_project_resources/test_production_runs/flows_prod_tests/output_flows_cont.hdf5 were run at a stage where the alpha_CE interpolation was incorrect - I think the interpolator was set up over non-log alpha_CE but then given alpha_CE for interpolation (commit ef80d32932cbdad44a5aa9d22d3091160073d4b0). These flow models also had CE retrained for a discrete inference. But the previous previous results in /AMAZE_project_rescources/continuous_GWTC-3/ I'm not sure why they also rail at alpha_CE=5. 

Checking old flow models which look like they trained slightly better (checked in /data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/notebooks/plot_obs_likelihoods.ipynb). Stopped the KDE models which hadn't started running yet and training a new flow model in inputs/prod_CEtrain.

28-11-24

Resubmitted the continuous and discrete inference with the mixed_flows input normalising flows, this and KDE inference is running with code version 6aa3b0f020f02bb42b16c993e8d39778e25a1f6b on flows_prod branch. The continuous inference outputs are in dir Nflows_AMAZE_paper/outputs/cont_GWTC3/prod_retrainedCE and the discrete inference is in Nflows_AMAZE_paper/outputs/discrete_GWTC3/flow_retrainedCE. APart from the different CE flow, the other flows are the same.
The KDEs were running with multi_proc on the GPUs, but I'm not sure if this was actually using the GPUs or not, so I removed 3/5 slower jobs and resubmitted on CPU so that the flow jobs can utilise CPU. The GPU jobs are seed 12 and 94 and the others are running on CPU. It's a race to see who is faster.

9-12-24

Checking up on runs, only waiting on 2/5 KDE jobs, but they got evicted and are starrting over after almost being done *upside-down smiley*. I don;t have capacity right now to get them to run on multiple cores.

Going to make plots in plots/prod without these KDE models, so I can update my paper plots.

15-1-25

Welcome to the new year. 

Finished results are in:
Nflows_AMAZE_paper/outputs/cont_GWTC3/prod_retrainedCE
Nflows_AMAZE_paper/outputs/discrete_GWTC3/flow_retrainedCE
Nflows_AMAZE_paper/outputs/discrete_GWTC3/KDEs

All inference is finished, most of the plots are made. Deciding what population to make in Fig. 1, current one is chib=0, alphaCE=2.

I was reminded that Tex doesn't work with the condor submission, so plot generation is by generating results on condor and then plotting just with bash on terminal, with --justplot.

Remade llh_ratio plot in prod_091224 with mixed_models and new contours, though labels need fixing.
Also ran calculate KLS with 10000 points from the training samples (10000 p_flow,p_KDE values), saved values of the flow||model and KDE||model to the Nflows_AMAZE_paper/plots/prod_091224/data directory, then saved these in the macros notebook too.
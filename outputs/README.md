18-11-24

Checking in on my tests (250 walkers and 1000 steps per walker), the continuous flows run is at 90% done after running for 6 days. Unfortunately I haven't saved the burnin for these, so will check each chain after and check if they have converged already or need to increase the burnin fraction.

The test KDE run is going to take forver, it's freezing probabilities for model 5/36 after 3 days. This will not depend on the number of walkers so it's just number of posterior samples that needs reduced. My optimisation in /notebooks/plot_obs_likelihood.ipynb suggested 11058 samples, so I will delete the current run and try that.

Otherwise setting up a run to train the CE model with a test population model removed. The outputs from this will live in /CE_test_population, and flow models in /inputs/CE_test_population. Running this with bf0bdd13f746facbdd6bc04737342312fbd3bd78 code ver.

19-11-24

Checked burnin for test run at /data/wiay/2297403c/amaze_model_select/Nflows_AMAZE_paper/outputs/cont_GWTC3/test_121124/plot_chains.ipynb
There is some burnin still present here, I think I will increase the burnin fraction from 0.2 (the test run) to 0.4 for the production runs.

The CE test smdl run also finished and I plotted the trained flow on the test model, it looks good.

I think I will use the trained flows I have already for inference, just copy over the test_121124 models, as I haven't made any changes to the training since running these tests.
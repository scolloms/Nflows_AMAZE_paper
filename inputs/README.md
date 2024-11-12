08-10-24 Storm Colloms

Created repo with inputs folder, created gw_names.txt with names of events used in GWTC-3 analyses. This is copied from an output of April's notebook, which I've checked 
(see AMAZE_project_resources/gwtc3_events/event_names.ipynb) contains the same list of events as the processed events in April's data release. Will use these events to generate processed events for runs.

22-10-24 Storm Colloms

Yesterday I tried regenerating GWTC 2.1 and GWTC 3 events based on notebook
/data/wiay/2297403c/amaze_model_select/AMAZE_model_selection/notebooks/process_GWTCdata_flows.ipynb

This was fine for the majority of the events with analytic priors but 2 events both had one sample at dL < minimum prior dL of 100 Mpc.
The interpolation was fine, but the prior ranges associated with the samples were incorrect.

The two events were GW191216_213338 and GW200225_060421 which had samples around 87, 88 Mpc.

Now the min dL for the prior on these events is the minimum dL in their samples.

Regenrating the events according to this notebook at git hash:
c4f2d4c3e1a48c67151c5ca58ddcbbe0e2ff5485

Honestly unsure about whether to git track the input data files, leaving untracked for now.

23-10-24 Storm Colloms

Copied over simulated events processed file that I originally downloaded from April's zenodo. The subsets of events (e.g. _50events, _100events)
contain the first 50, 100 events from the total simulated population corresponding to the correct model.

Created flow_models dir in inputs, this will be the saved location of the trained flows.

06-11-24 Storm Colloms

Confirmed that cosmo weighted events are fine to use with uniform in source frame redshift prior.
Trying pre-processing IMRPhenom samples, which is overwriting the priors files rn. These are stored in events-IMR (currently .gitignored) and were generated with a3f89f7f9c04eb1399ed9bf04abddc111d5e166c version of notebook.

GWTC-3/events directory is total 121MB which is trackable via git-lfs if I want to set this up.
Installed git lfs and tracking GWTC-3/events dir with this. current git attribites file set to track any 
*.hdf5 files with lfs.
I think the priors files should actually be the same, as they are both read from the ['C01:IMRPhenomXPHM']['priors']['analytic'] key. Anyway this just changes the max/min ranges for the prior, which is just a prior scaling on the priors.
Commiting priors files to repo.

23-11-24

Started regenerating mixed samples just to check i didn't overwrite the current inputs with nocosmo samples at any point, but I think I started generating these in another folder and then removed this. Anyway the test samples (generated correctly) looked the same as whats in the current inputs/GWTC-3/events so we are good. I did delete the test folder. Learning to take good notes is a process.

Going to set up a *test* continuous inference run in this directory, after having decided the ordeal with the number of samples for undersampled events.
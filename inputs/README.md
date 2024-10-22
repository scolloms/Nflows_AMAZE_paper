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
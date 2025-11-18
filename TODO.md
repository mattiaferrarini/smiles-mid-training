# TODO

At this point, everything should, at least, start running following the instructions in ``README.md``.
Below is what is left to do for this week. There are many things we could do to improve our code, but I would start from the necessary ones to have a working solution. Once that is done, we can "have fun" improving the rest.

## Must Do
Everything needs to be done on the ``training\training.py`` script. The corresponding SLURM job probably does not need changes. 

In order of decreasing priority:
1. Add WandB logging, ideally with a token added to ``.env``.
2. Check with jobreport that:
    1. We are using GPUs and not CPUs;
    2. We are using multiple nodes and GPUs per node.
3. Find out what we should do with the slurm parameters listed in ``README.md`` (look online).
4. (Optional but nice) Tune hyperparameters to use available resources as well as possible.
5. (Optional but nice) Tune hyperparameters so that we use sensible values for the training (e.g. look online what are standard values for learnign rate, weight decay, ...).
6. (Just optional) Following same idea of previous point: maybe change save strategy and similar stuff.


= Summary
This repository contains all code and data needed to replicate our project (not counting the real robot experiments of course, you need robots for that). Below, we write multiple sections to help explain the repo.

== Dependencies
See `pip_freeze.txt` for the dependencies used to run our experiments. You should be able to run `pip install -r pip_freeze.txt` to replicate our setup.

== Datasets
We encode our real-world datasets as h5 files. See `dataset.h5` for the dataset we used for most of training. `dataset-X.h5` contains X% of our dataset, used for the data size ablation study. For example `dataset-50.h5` corresponds to half the size of the original dataset. The original CSVs we used to generate the dataset are also included in `data/robomaster_collect*`. We generate natural language tasks using the `rewards.py`, `rewards2.py`, and `tasks.py` files.

== Losses
`losses.py` contains our implementation of all the objectives we used.

== Dynamics Simulator
We trained a dynamics simulator on the dataset. See `data/dynamics_model_weights.eqx` for the weights, and `dynamics_model.py` for the model. `train_dynamics_model.py` will actually train the model. You can run `play_dynamics_model.py` to play with the trained dynamics model using `pygame`, note that the WASD controls may be a bit unintuitive.

== LLM Analysis
Simply run `validate_llm.py` to rerun our experiment that evaluted the latent space of LLMs. It will log the various loss curves to `wandb`.

== Policy Training
`run_jax_offline_marl.py` is the main training script we used to train all of our policies. It relies on the experiment configurations located in `experiments/*.yaml`. `benchmark_policy.py` prints out how long it takes to execute a policy on your machine.

== Policy Evaluation
See `evaluate_policy.py` for evaluating trained policies in simulation. In the real-world, the scripts `robomaster_control_*` are used for executing robomasters in the real world. In particular, `robomaster_control_nodes` contains the `ros2` nodes used for dataset collection and training.
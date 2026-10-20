# SA-Scheduling

This repository provides sustainability-aware scheduling code for self-driving laboratories, including reinforcement learning (RL) for initial scheduling, constraint programming (CP) for post-optimization, a hybrid scheduling system, dynamic simulation, and paper plotting tools.

| Directory | Description |
| --- | --- |
| `algorithm/` | Scheduling algorithms, including RL and CP post-optimization. |
| `multi_pms/` | Scheduling services, data processing, dynamic simulation, and simulation and experimental results used in the paper. |
| `rl/` | PPO + HGNN training framework and scheduling environment. |

## Installation

Use Python 3.10. Create and activate a virtual environment from the repository root, then install the dependencies.

```bash
python3.10 -m venv SA-Scheduling
source SA-Scheduling/bin/activate
python -m pip install -r requirements.txt
```

## Quick Start

Run the following commands from the repository root with the virtual environment activated.

### 1. Dynamic Simulation and Paper Figures

#### Generate Paper Figures

Generate the paper figures from the existing data. Figures are saved to `multi_pms/expr_data/metrics/`.

```bash
python multi_pms/expr_data/paper_plot.py
```

#### Reproduce the Paper Results

Open two terminals and activate the virtual environment in each. Start the server first, then the simulation client.

Terminal 1 (server):

```bash
python multi_pms/performance_test/dynamic_scheduling_service.py --debug_flask_port 5082 --database_file dynamic_replay_paper --schedule_method rl_release_or --schedule_time_limit 300
```

Keep this server running with the 300 s solver budget. The paper used 256 CP threads; pass `--threads 256` when starting the server on matching hardware.

Terminal 2 (client): run the following experiment groups sequentially. They all use the same isolated server and database. Simulation results are saved to `multi_pms/expr_data/data/`.

**Fig. 3a–e and Supplementary Fig. S1:** run the main configuration, covering five 36-task scenarios, peak/off-peak price ratios of 1:1–4:1, and seven time-value settings. The methods are completion-time-only CP, sustainability-aware scheduling, and simulated manual scheduling. These simulations also supply the simulated ranges in the Hefei panel of Supplementary Fig. S3.

```bash
python multi_pms/performance_test/replay_experiment.py --config multi_pms/expr_data/data/replay_experiment_config.json --dynamic-schedule-port 5082 --dynamic-db-path multi_pms/database/dynamic_replay_paper.sqlite
```

Summary and detailed records are appended to `multi_pms/expr_data/data/jsonl/metrics_dynamic.jsonl` and `metrics_dynamic_details.jsonl`.

**Fig. 3f and Supplementary Fig. S2:** enable `uniqueCompare` to compare the three paper configurations across five scenarios with 6–12 concurrent tasks. The objective-matching CP configuration uses the corresponding RL-initialized result as its target and has an extended 1,800 s limit.

```bash
python multi_pms/performance_test/replay_experiment.py --config multi_pms/expr_data/data/replay_experiment_config.json --unique-compare --dynamic-schedule-port 5082 --dynamic-db-path multi_pms/database/dynamic_replay_paper.sqlite
```

Summary and detailed records are appended to `multi_pms/expr_data/data/jsonl/metrics_dynamic_unique_task.jsonl` and `metrics_dynamic_unique_task_details.jsonl`.

**Fig. 5a and Supplementary Fig. S3, overseas tariff cases:** run the Liverpool, Berkeley, Toronto, and Daejeon configurations. The output filename below matches the plotting script, including the existing spelling `metircs`.

```bash
for country in uk_liverpool us_berkeley ca_toronto kr_daejeon; do
  python multi_pms/performance_test/replay_experiment.py \
    --config "multi_pms/expr_data/data/replay_${country}_config.json" \
    --dynamic-schedule-port 5082 \
    --dynamic-db-path multi_pms/database/dynamic_replay_paper.sqlite \
    --results-file metircs_dynamic_countries.jsonl || break
done
```

Summary and detailed records are appended to `multi_pms/expr_data/data/jsonl/metircs_dynamic_countries.jsonl` and `metircs_dynamic_countries_details.jsonl`.

After the experiments finish, regenerate the paper figures:

```bash
python multi_pms/expr_data/paper_plot.py
```

To redraw the figures from the supplied result data, run only this plotting command. New algorithm runs can differ numerically because RL inference samples schedules and CP search is time-limited.

The scheduling service entry point is [`multi_pms/scheduling.py`](multi_pms/scheduling.py#L1). External systems can submit tasks and equipment states through its HTTP interface and receive scheduling results using the expected input and output formats. Use [`multi_pms/fake_dms.py`](multi_pms/fake_dms.py#L1) to simulate DMS interaction with the scheduling service.

### 2. RL Training

Train a PPO + HGNN policy to generate initial schedules that satisfy experimental constraints.

From the repository root with the virtual environment activated, run:

```bash
cd rl && SWANLAB_MODE=offline python train.py --num_jobs 10 --num_machines 20
```

#### Data and Parameters

Start training from `rl/`. Training loads the corresponding instance size from `data/data_train/` and `data/data_val/`. Available datasets are `0304` (3 jobs, 4 stations) and `1020` (10 jobs, 20 stations).

Training options are defined in the [argument parser](rl/utils/args.py#L27).

For example, train on the 0304 dataset from `rl/`:

```bash
SWANLAB_MODE=offline python train.py --num_jobs 3 --num_machines 4
```

#### Training Output

The [training entry point](rl/train.py#L36) records offline logs with SwanLab, validates automatically during training, and saves models when validation performance improves. Models are saved under `rl/data/runs/` in directories named by instance size and run time.

## License

The project code is released under the [MIT License](LICENSE). Third-party map materials retain their respective licensing terms, documented in the [map asset credits](multi_pms/expr_data/assets/README.md).

## Map Sources and Acknowledgements

The Fig. 5 map uses the following sources:

| Material | Source | Use in this project |
| --- | --- | --- |
| World map background | [World map blank without borders.svg](https://commons.wikimedia.org/wiki/File:World_map_blank_without_borders.svg), by Crates, Wikimedia Commons; used under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). | Rasterized background for the global SDL map. |
| Self-driving laboratory locations | [The Global Self-Driving Lab Landscape](https://github.com/yetiswang/sdl-map), by yetiswang and contributors. | Location reference for the SDL markers. |

See the [map asset credits](multi_pms/expr_data/assets/README.md) for local filenames, adaptations, and upstream licensing information. When reusing the map panel in a paper or presentation, retain the source credits and the CC BY-SA 4.0 notice for the map-derived material.

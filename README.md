# FastSeqTrack and FasterSeqTrack

**Exploring Parallel and Dynamic Sequence Learning for Efficient UAV Tracking**.

FastSeqTrack improves the efficiency of sequence-learning-based tracking through parallel coordinate prediction. FasterSeqTrack further introduces confidence-based early exits, allowing easy frames to terminate at intermediate encoder stages while retaining deeper computation for challenging frames.

## Highlights

- **Parallel sequence prediction:** FastSeqTrack predicts the four bounding-box coordinates in one forward pass instead of autoregressive decoder execution.
- **Encoder-refined tracking tokens:** Four coordinate-specific tracking tokens are refined by the encoder and used for bounding-box prediction.
- **Dynamic early exiting:** FasterSeqTrack can exit after intermediate encoder stages according to prediction confidence, reducing unnecessary encoder and decoder computation.
- **Multiple operating points:** The repository provides one-template Lite models and two-template models for different accuracy-efficiency requirements.

## Repository Structure

```text
FastSeqTrack/
|-- FastSeqTrack/        # Parallel sequence prediction tracker
|-- FasterSeqTrack/      # Dynamic early-exit tracker
`-- README.md
```

The available configurations are:

| Tracker | Configuration | Templates |
|---|---|---:|
| FastSeqTrack-Lite | `fastseqtrack_b256_1t` | 1 |
| FastSeqTrack | `fastseqtrack_b256_2t` | 2 |
| FasterSeqTrack-Lite | `fasterseqtrack_b256_1t` | 1 |
| FasterSeqTrack | `fasterseqtrack_b256_2t` | 2 |

## Installation

The code is developed with Python 3.8 and PyTorch. Install a PyTorch build compatible with your CUDA environment first, and then install the remaining dependencies.

```bash
conda create -n fastseqtrack python=3.10 -y
conda activate fastseqtrack


Clone the repository and enter its root directory:

```bash
git clone https://github.com/vision4drones/FastSeqTrack.git
cd FastSeqTrack
```

## Data Preparation

The training configurations use LaSOT, GOT-10k, COCO, TrackingNet, VastTrack, VisDrone, and WebUAV-3M. The default path generator assumes the following layout:

```text
data/
|-- lasot/
|-- got10k/
|   `-- train/
|-- coco/
|-- trackingnet/
|-- vasttrack/
|-- visdrone/
|-- webuav3m/
|   |-- trainval/
|-- UAV123/
|-- DTB70/
|-- uavdt/
|-- VisDrone2018-SOT-test-dev/
`-- UAVTrack112/
```

For FastSeqTrack:

```bash
cd FastSeqTrack
python tracking/create_default_local_file.py \
    --workspace_dir . \
    --data_dir ./data \
    --save_dir .
```

For FasterSeqTrack:

```bash
cd ../FasterSeqTrack
python tracking/create_default_local_file.py \
    --workspace_dir . \
    --data_dir ./data \
    --save_dir .
```

The generated files are:

```text
lib/train/admin/local.py          # Training dataset and output paths
lib/test/evaluation/local.py      # Testing dataset, checkpoint, and result paths
```

The `--data_dir` argument is interpreted independently inside each implementation directory. You may instead provide an absolute dataset path, for example `--data_dir /path/to/data`.

## Pretrained Models

Place the trained checkpoints in the paths expected by the testing code:

```text
FastSeqTrack/checkpoints/train/fastseqtrack/
|-- fastseqtrack_b256_1t/FASTSEQTRACK_ep0500.pth.tar
`-- fastseqtrack_b256_2t/FASTSEQTRACK_ep0500.pth.tar

FasterSeqTrack/checkpoints/train/fasterseqtrack/
|-- fasterseqtrack_b256_1t/FASTERSEQTRACK_ep0500.pth.tar
`-- fasterseqtrack_b256_2t/FASTERSEQTRACK_ep0500.pth.tar
```

Pretrained model and raw-result download links will be provided here.

## Training

Run each command from the corresponding implementation directory.

### FastSeqTrack

```bash
cd FastSeqTrack

# FastSeqTrack-Lite, one template
torchrun --nproc_per_node=8 lib/train/run_training.py \
    --script fastseqtrack \
    --config fastseqtrack_b256_1t \
    --save_dir .

# FastSeqTrack, two templates
torchrun --nproc_per_node=8 lib/train/run_training.py \
    --script fastseqtrack \
    --config fastseqtrack_b256_2t \
    --save_dir .
```

### FasterSeqTrack

```bash
cd FasterSeqTrack

# FasterSeqTrack-Lite, one template
torchrun --nproc_per_node=8 lib/train/run_training.py \
    --script fasterseqtrack \
    --config fasterseqtrack_b256_1t \
    --save_dir .

# FasterSeqTrack, two templates
torchrun --nproc_per_node=8 lib/train/run_training.py \
    --script fasterseqtrack \
    --config fasterseqtrack_b256_2t \
    --save_dir .
```

Change `--nproc_per_node` according to the number of available GPUs. Checkpoints and training logs are saved under `checkpoints/train/<tracker>/<config>/` and `logs/`, respectively.

## Testing

Run the following commands from either `FastSeqTrack/` or `FasterSeqTrack/`.

### FastSeqTrack Examples

```bash
cd FastSeqTrack

# FastSeqTrack on VisDrone2018
python tracking/test.py fastseqtrack fastseqtrack_b256_2t \
    --dataset_name visdrone \
    --threads 2 \

# FastSeqTrack-Lite on UAV123
python tracking/test.py fastseqtrack fastseqtrack_b256_1t \
    --dataset_name uav \
    --threads 2 \
```

### FasterSeqTrack Examples

```bash
cd FasterSeqTrack

# FasterSeqTrack on DTB70
python tracking/test.py fasterseqtrack fasterseqtrack_b256_2t \
    --dataset_name dtb70 \
    --threads 2 \

# FasterSeqTrack-Lite on UAVTrack112
python tracking/test.py fasterseqtrack fasterseqtrack_b256_1t \
    --dataset_name uavtrack112 \
    --threads 2 \
```

Supported UAV benchmark identifiers include:

| Benchmark | `--dataset_name` |
|---|---|
| UAV123 | `uav` |
| DTB70 | `dtb70` |
| UAVDT | `uavdt` |
| VisDrone2018 | `visdrone` |
| UAVTrack112 | `uavtrack112` |
| UAVTrack112-L | `uavtrack` |

To test a single sequence, additionally provide `--sequence <sequence_name_or_index>`.

## Evaluation

Tracking results are saved under:

```text
test/tracking_results/<tracker>/<config>/
```

Set the tracker, configuration, and dataset in `tracking/analysis_results.py`, and then run:

```bash
python tracking/analysis_results.py
```

## Model Profiling

The repository provides profiling scripts based on THOP.

For FastSeqTrack:

```bash
cd FastSeqTrack
python tracking/profile_model.py \
    --script fastseqtrack \
    --config fastseqtrack_b256_1t
```

For FasterSeqTrack, separate scripts are provided for fixed encoder exits and complete encoder-decoder execution:

```bash
cd FasterSeqTrack

# One-template model exiting at encoder layer 6
python tracking/profile_model_1t_exit.py \
    --config fasterseqtrack_b256_1t \
    --exit_layer 6 \
    --threshold 0

# One-template model with complete encoder-decoder execution
python tracking/profile_model_1t_de.py \
    --config fasterseqtrack_b256_1t

# Two-template model exiting at encoder layer 12
python tracking/profile_model_2t_exit.py \
    --config fasterseqtrack_b256_2t \
    --exit_layer 12 \
    --threshold 0

# Two-template model with complete encoder-decoder execution
python tracking/profile_model_2t_de.py \
    --config fasterseqtrack_b256_2t
```

The default early-exit confidence threshold used by FasterSeqTrack is configured through `TEST.SKIP_THRESHOLD` in the corresponding YAML file.

## Acknowledgement

This repository is built upon [SeqTrack](https://github.com/microsoft/VideoX/tree/master/SeqTrack). The codebase also benefits from STARK, PyTracking, timm, and MAE. We thank the authors of these projects for making their work publicly available.

## Citation

If this repository is useful for your research, please cite our paper.

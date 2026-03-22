# Vietnamese Speech Recognition with Whisper-Tiny

Fine-tuning Whisper (openai/whisper-tiny) for **Vietnamese Automatic
Speech Recognition (ASR)** using Hugging Face Transformers.

This project improves transcription accuracy by training on a Vietnamese
speech dataset.

------------------------------------------------------------------------

## Features

-   Fine-tune Whisper for Vietnamese ASR\
-   Evaluate performance using **Word Error Rate (WER)**\
-   Run inference on custom audio files\
-   GPU acceleration support (CUDA)

------------------------------------------------------------------------

## Dataset

-   Dataset: `doof-ferb/infore1_25hours`

-   Automatically downloaded via Hugging Face:

``` python
from datasets import load_dataset

dataset = load_dataset("doof-ferb/infore1_25hours")
```

------------------------------------------------------------------------

## Installation

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Training

Run the training script:

``` bash
python src/train.py
```

-   Model checkpoints will be saved in:

```{=html}
<!-- -->
```
    ./model/

------------------------------------------------------------------------

## Inference (Speech-to-Text)

1.  Place your audio file in the `data/` folder\
    *(e.g., `test.mp3`)*

2.  Run:

``` bash
python src/inference.py
```

### Example

    Input: test.mp3  
    Output: Trời hôm nay thật đẹp.

------------------------------------------------------------------------

## Evaluation

-   Metric: **Word Error Rate (WER)**

------------------------------------------------------------------------

## Model

-   Base model: `openai/whisper-tiny`\
-   Fine-tuned checkpoints:

```{=html}
<!-- -->
```
    ./model/checkpoint-xxx

------------------------------------------------------------------------

## Notes

-   Audio is resampled to **16kHz** for Whisper compatibility\
-   Training uses **mixed precision (fp16)**\
-   GPU is recommended for faster training

------------------------------------------------------------------------

## Project Structure

    ├── data/               # Input audio files
    ├── model/              # Trained checkpoints
    ├── src/
    │   ├── train.py        # Training script
    │   └── inference.py    # Inference script
    ├── requirements.txt
    └── README.md


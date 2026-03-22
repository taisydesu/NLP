Fine-Tuning Whisper-Tiny for Vietnamese Speech Recognition

This project builds a Vietnamese Automatic Speech Recognition (ASR) system using the Whisper model (openai/whisper-tiny) from Hugging Face Transformers.

The model is fine-tuned on a Vietnamese speech dataset to improve transcription accuracy.

## Features
Fine-tune Whisper for Vietnamese speech recognition
Evaluate model using Word Error Rate (WER)
Perform inference on custom audio files
Supports GPU acceleration (CUDA)

## Dataset
# Dataset used:
doof-ferb/infore1_25hours

# Download is handled automatically using Hugging Face:
from datasets import load_dataset
dataset = load_dataset("doof-ferb/infore1_25hours")

## Installation
pip install -r requirements.txt

## Training
# Run:
python src/train.py

# Model checkpoints will be saved in:
./model/

## Inference (Speech-to-Text)
# Place your audio file in data/ folder (e.g., test.mp3)
Run:
python src/inference.py

# Example Output
Input: test.mp3
Output: Trời hôm nay thật đẹp.

## Evaluation Metric
Word Error Rate (WER)

## Notes
Audio is resampled to 16kHz for Whisper compatibility
Training uses mixed precision (fp16)
GPU recommended for faster training

## Model
# Base model:
openai/whisper-tiny

# Fine-tuned checkpoints:
./model/checkpoint-xxx
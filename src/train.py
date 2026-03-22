import os
from datasets import load_dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from preprocess import prepare_dataset, cast_dataset
from collator import DataCollatorSpeechSeq2SeqWithPadding
from metrics import compute_metrics


def main():
    import torch
    print("GPU available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    model_id = "openai/whisper-tiny"

    dataset = load_dataset("doof-ferb/infore1_25hours", split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    dataset = cast_dataset(dataset)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_id, language="vietnamese", task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        model_id, language="vietnamese", task="transcribe"
    )

    dataset = dataset.map(
        lambda x: prepare_dataset(x, feature_extractor, tokenizer),
        remove_columns=dataset["train"].column_names,
        num_proc=2
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir="./model",
        per_device_train_batch_size=16,
        learning_rate=1e-5,
        max_steps=500,
        fp16=True,
        eval_strategy="steps",
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    trainer.train()


if __name__ == "__main__":
    main()
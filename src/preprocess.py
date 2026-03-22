from datasets import Audio

def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = tokenizer(batch["transcription"]).input_ids

    return batch


def cast_dataset(dataset):
    return dataset.cast_column("audio", Audio(sampling_rate=16000))
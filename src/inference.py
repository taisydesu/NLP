import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration


def load_pipeline(model_path):
    base_model = "openai/whisper-tiny"

    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(base_model)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=0 if torch.cuda.is_available() else -1,
    )

    return pipe


def transcribe(audio_path, pipe):
    result = pipe(
        audio_path,
        generate_kwargs={"language": "vietnamese", "task": "transcribe"}
    )
    return result["text"]


if __name__ == "__main__":
    pipe = load_pipeline("./model/checkpoint-500")
    print(transcribe("data/test.mp3", pipe))
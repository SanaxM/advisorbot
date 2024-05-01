import os
import openai
import dotenv

import librosa
import torch
import logging
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline
from datasets import load_dataset
import soundfile as sf

logging.getLogger("transformers").setLevel(logging.ERROR)

# Load STT
tokenizer_stt = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model_stt = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load TTS
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

# speaker embeddings dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Load speech audio - Question asked: Does the length of credit history matter?
speech_stt, rate_stt = librosa.load("credit-history-stt.wav", sr=16000)

# Tokenize speech input
input_values_stt = tokenizer_stt(speech_stt, return_tensors="pt").input_values

# Store logits (non-normalized predictions)
logits_stt = model_stt(input_values_stt).logits

# Store predicted ids
predicted_ids_stt = torch.argmax(logits_stt, dim=-1)

# Decode the audio to generate text
transcription_stt = tokenizer_stt.decode(predicted_ids_stt[0])

print("Transcription of Input Audio:", transcription_stt)

dotenv.load_dotenv()

endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_ID")

client = openai.AzureOpenAI(
    base_url=f"{endpoint}/openai/deployments/{deployment}/extensions",
    api_key=api_key,
    api_version="2023-08-01-preview",
)

completion = client.chat.completions.create(
    model=deployment,
    messages=[
        {
            "role": "user",
            "content": transcription_stt,
        },
    ],
    extra_body={
        "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": os.environ["AZURE_AI_SEARCH_ENDPOINT"],
                    "indexName": os.environ["AZURE_AI_SEARCH_INDEX"],
                    "semanticConfiguration": "default",
                    "queryType": "simple",
                    "fieldsMapping": {},
                    "inScope": True,
                    "roleInformation": "You are an AI assistant that helps people find information.",
                    "filter": "",
                    "strictness": 5,
                    "topNDocuments": 5,
                    "key": os.environ["AZURE_AI_SEARCH_API_KEY"],
                }
            }
        ]
    }
)

# Do the TTS for the AI output
revised_speech_1 = completion.choices[0].message.content.replace("[doc1]", "")
revised_speech_2 = revised_speech_1.replace("[doc2]", "")
speech_input = revised_speech_2
speech = synthesiser(speech_input, forward_params={"speaker_embeddings": speaker_embedding})

# Save speech audio
sf.write("credit-history-length.wav", speech["audio"], samplerate=speech["sampling_rate"])

print(speech_input)

from openai import OpenAI
from elevenlabs import ElevenLabs
from elevenlabs import play
import requests
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
el_client = ElevenLabs(api_key=os.environ.get("ELEVEN_LABS_API_KEY"))


class ExtractedText(BaseModel):
    extracted_text: str
    page_numbers: list[str]


class RGB(BaseModel):
    red: int
    green: int
    blue: int


class SentimentAnalysis(BaseModel):
    temperature: float
    vibration_amplitude: int
    vibration_frequency: float
    rgb: RGB


def extract_text_from_image(base64_image: str):
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts text from books. Extract the text from the pages of the book specifically. If the text is cutoff, extract it as normal. Don't miss any text. Don't make up any sentences.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the text from the pages of the book.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        response_format=ExtractedText,
    )

    return json.loads(response.choices[0].message.content)


def chunk_text(text):
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    temp_chunk = ""

    for sentence in sentences:
        temp_chunk += sentence.strip() + ". "
        if len(temp_chunk.split()) >= 5:  # Create chunks of at least 5 words
            chunks.append(temp_chunk.strip())
            temp_chunk = ""

    if temp_chunk:
        chunks.append(temp_chunk.strip())

    return chunks


def analyze_mood(chunk: str):
    prompt = f"""
    Analyze the mood of the following text and provide a valid response in the format below.
    Provide the temperature, RGB, and other parameters as needed.

    Temperature: -1.0 to 1.0 (-1 is hottest, 1 is coldest).
    Vibration Amplitude (Intensity): 0 - 127.
    Vibration Frequency: Typically ranges from 100 Hz to 250 Hz based on provided samples.
    RGB: (0, 0, 0) to (255, 255, 255).
    
    Text: "{chunk}"
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment analysis assistant that helps people that are visually impaired feel immersive haptics based on text. The text is from a page of a book. Always respond in valid format.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=SentimentAnalysis,
    )
    return json.loads(response.choices[0].message.content)


def convert_text_to_speech(text, filename):
    audio_generator = el_client.text_to_speech.convert(
        text=text,
        voice_id=os.environ.get("ELEVEN_LABS_VOICE_ID"),
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    with open(f"./audio/{filename}", "wb") as f:
        for chunk in audio_generator:
            f.write(chunk)

    print(f"Audio saved: {filename}")

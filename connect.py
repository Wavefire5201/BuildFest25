from openai import OpenAI
import requests
import base64
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()


# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ==============================
# 1. Extract Text from Image
# ==============================
def extract_text_from_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract and return the text from this image:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content


# ==============================
# 2. Chunk Text
# ==============================
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


# ==============================
# 3. Analyze Mood & Generate RGB
# ==============================
def analyze_mood(chunk):
    prompt = f"""
    Analyze the mood of the following text and provide a temperature between -1 (very hot) and 1 (very cold).
    Then, convert the temperature into an RGB color in (R,G,B) format.

    Text: "{chunk}"

    Output format: {{ "temperature": <value>, "rgb": "(R,G,B)" }}
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",  # Use the latest model
        messages=[
            {
                "role": "system",
                "content": "You are a sentiment analysis assistant. Always respond in valid JSON format.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_object"
        },  # Ensure the response is in JSON format
    )
    response_text = response.choices[0].message.content
    try:
        analysis = json.loads(response_text)
        return analysis["temperature"], analysis["rgb"]
    except json.JSONDecodeError:
        print("Error parsing JSON:", response_text)
        return None, None


# ==============================
# 4. Convert Text to Speech (TTS)
# ==============================
def convert_text_to_speech(text, filename):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{os.environ.get("ELEVEN_LABS_VOICE_ID")}"
    headers = {"xi-api-key": os.environ.get("ELEVEN_LABS_API_KEY")}
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Audio saved: {filename}")
    else:
        print(f"Error generating TTS: {response.status_code}")


# ==============================
# Main Function
# ==============================
def main(image_path):
    text = extract_text_from_image(image_path)
    if not text:
        print("No text extracted.")
        return

    print("Extracted Text:\n", text)

    chunks = chunk_text(text)
    for idx, chunk in enumerate(chunks, start=1):
        temperature, rgb = analyze_mood(chunk)
        if temperature is not None and rgb is not None:
            print(f"Chunk {idx}: {chunk}")
            print(f"Temperature: {temperature}, RGB: {rgb}")

            output_filename = f"chunk_{idx}.mp3"
            convert_text_to_speech(chunk, output_filename)


if __name__ == "__main__":
    main("Casting_4XKEjyHtxs.png")

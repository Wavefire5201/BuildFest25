from PIL import ImageGrab, Image, ImageFilter
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import time

load_dotenv()

time.sleep(5)


class ExtractedText(BaseModel):
    extracted_text: str
    page_numbers: list[str]


# Capture the screenshot
screenshot = ImageGrab.grab()

# Apply image enhancements for better text clarity
screenshot = screenshot.filter(ImageFilter.SHARPEN)  # Sharpen the image
screenshot = screenshot.filter(ImageFilter.DETAIL)  # Enhance details
screenshot = screenshot.filter(
    ImageFilter.EDGE_ENHANCE
)  # Enhance edges for better text definition
screenshot.show()
import base64
from io import BytesIO

buffered = BytesIO()
screenshot.save(buffered, format="PNG")
base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
print(base64_image)

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that extracts text from books. Extract the text from the pages of the book specifically. If the text is cutoff, extract it as normal. Don't miss any text. Don't make up any sentences.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
    response_format=ExtractedText,
)

extracted_json = response.choices[0].message.content
print(response.choices[0].message)
print(extracted_json)



import google.generativeai as genai
from PIL import Image
import logging

def read_image_and_chunk(image_path: str, source_file: str):
    """
    Uses a multimodal model to describe an image and returns the description as a text chunk.
    """
    try:
        logging.info(f"Processing image file: {source_file} with multimodal vision.")
        # Use a vision-capable model
        model = genai.GenerativeModel('gemini-1.5-pro-latest') 
        
        image = Image.open(image_path)
        
        # The prompt asks the model to act as an OCR and data extractor
        prompt = "Describe all text and information visible in this image in detail. Extract any tables, charts, or key figures as structured text. Be precise and thorough."
        
        response = model.generate_content([prompt, image])
        
        if response.text:
            yield {
                "text": response.text,
                "source": f"Image content from '{source_file}'"
            }
            
    except Exception as e:
        logging.error(f"Failed to process image with vision model: {e}")
        return
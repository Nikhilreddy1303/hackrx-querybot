import google.generativeai as genai
from PIL import Image
import logging

def read_image_and_chunk(image_path: str, source_file: str):
    """
    Uses a multimodal model to perform OCR and data extraction on an image,
    returning only the raw transcribed text.
    """
    try:
        logging.info(f"Performing OCR and data extraction on: {source_file}")
        model = genai.GenerativeModel('gemini-1.5-pro-latest') 
        
        image = Image.open(image_path)
        
        # New, much stricter prompt to force data extraction instead of description.
        prompt = """You are a precision OCR and data extraction engine. Your only task is to transcribe the text from the provided slide image exactly as it appears.

**Instructions:**
1. Extract all text verbatim.
2. If you see a table, transcribe its content, using pipes (|) to separate columns.
3. Transcribe bullet points and lists as they are.
4. **Crucially, DO NOT add any commentary, explanations, or introductory sentences like 'This slide shows...' or 'The image contains...'.** Do not describe the slide's layout or colors. Only output the transcribed text.
"""
        
        response = model.generate_content([prompt, image])
        
        if response.text:
            yield {
                "text": response.text,
                "source": f"Content from '{source_file}'"
            }
            
    except Exception as e:
        logging.error(f"Failed to process image with vision model: {e}")
        return
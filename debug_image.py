import asyncio
import logging
import os  # <-- THE FIX IS HERE
from dotenv import load_dotenv
import google.generativeai as genai

# Import the functions we want to test
from read_url import download_file_from_url
from read_image import read_image_and_chunk

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

async def main():
    image_url = "https://hackrx.blob.core.windows.net/assets/Test%20/image.png?sv=2023-01-03&spr=https&st=2025-08-04T19%3A21%3A45Z&se=2026-08-05T19%3A21%3A00Z&sr=b&sp=r&sig=lAn5WYGN%2BUAH7mBtlwGG4REw5EwYfsBtPrPuB0b18M4%3D"
    
    print(f"--- Attempting to process image from URL: {image_url} ---")
    
    temp_path = None
    try:
        # Step 1: Download the file
        print("\n[1/3] Downloading image file...")
        temp_path, ext = download_file_from_url(image_url)
        print(f"      Success! Image downloaded to temporary path: {temp_path}")

        # Step 2: Process it with the multimodal parser
        print("\n[2/3] Sending image to Gemini for analysis...")
        chunks = list(read_image_and_chunk(temp_path, source_file="debug_image.png"))
        
        # Step 3: Print the result
        print("\n[3/3] Gemini Vision API Response:")
        if not chunks:
            print("      ---> FAILURE: No chunks were returned. The vision model likely failed silently.")
        else:
            print("      ---> SUCCESS! The model returned the following text chunk:")
            print("-" * 20)
            print(chunks[0]['text'])
            print("-" * 20)
            print(f"Source: {chunks[0]['source']}")

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        logging.error("The debug script failed.", exc_info=True)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    asyncio.run(main())
import requests
import logging
from google.generativeai.types import FunctionDeclaration, Tool

def make_http_get_request(url: str) -> str:
    """
    Makes an HTTP GET request to a specified URL and returns the response text.
    Use this to get information from web pages or specific API endpoints that use GET requests.
    """
    logging.info(f"AGENT TOOL USED: Making GET request to {url}")
    try:
        # Using a common user-agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # Truncate response to avoid overwhelming the context window
        return response.text[:8000]

    except requests.exceptions.RequestException as e:
        logging.error(f"Agent tool 'make_http_get_request' failed for URL {url}: {e}")
        return f"Error: Could not fetch data from the URL. Details: {e}"

# --- Tool Definition for Gemini ---
# This is how we describe our Python function to the Gemini model so it knows how and when to use it.

GET_REQUEST_TOOL = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="make_http_get_request",
            description="Makes an HTTP GET request to a URL and returns the text content. Essential for accessing web pages or APIs.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "url": {
                        "type": "STRING",
                        "description": "The complete, valid URL to fetch."
                    }
                },
                "required": ["url"],
            },
        )
    ]
)
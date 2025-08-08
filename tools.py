import requests
import logging

def make_http_get_request(url: str) -> str:
    """
    Makes an HTTP GET request to the specified URL and returns the response text.
    This is a tool for the AI agent to use.
    """
    try:
        logging.info(f"AGENT TOOL USED: Making GET request to {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Agent tool failed for URL {url}: {e}")
        return f"Error: Could not fetch data from the URL. {e}"
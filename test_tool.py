# test_tool.py
import logging
from agent_tools import make_http_get_request

# Configure basic logging to see output
logging.basicConfig(level=logging.INFO)

# The URL from the "Flight Number" PDF to get the city
test_url = "https://register.hackrx.in/submissions/myFavouriteCity"

print(f"--- Testing Agent Tool ---")
result = make_http_get_request(test_url)
print(f"Result from tool: {result}")
print(f"--- Test Complete ---")
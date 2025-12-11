import requests
import concurrent.futures
import json
import random
import urllib3
import time

# Suppress SSL warnings (only needed if verification is disabled)
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configurable parameters
NUM_WORKERS = 200  # concurrent threads
NUM_REQUESTS = 5000  # total API flows to trigger
EMBED_DIM = 3584  # match your backend expectation
REQUEST_DELAY = 0.02  # Delay between requests in seconds (to avoid rate limiting)

# SSL Certificate path for verification
SSL_CERT_PATH = None
# Set to False to disable SSL verification (not recommended)
VERIFY_SSL = False  # Use certificate file for verification

# Base URL - Check what api_manager actually uses
# The api_manager resolves URL from: env var > Pulumi config > default
# Default in api_manager is: "http://35.231.234.99"
# If you're using 34.74.208.58, it might be from Pulumi config or env var
# Try the default first, or check your .pulumi-config.json file
BASE_URL = "http://35.231.234.99"  # Default from api_manager - change if different
TEXT_EMBED_URL = f"{BASE_URL}/embed-text"
PREDICTION_URL = f"{BASE_URL}/predict"

# Sample data matching api_manager format
SAMPLE_TEXT_DESCRIPTION = "Red, itchy rash on the face with small bumps"
SAMPLE_VISION_EMBEDDING = [random.uniform(-1, 1) for _ in range(EMBED_DIM)]

# Reuse a session to avoid reconnect overhead
# Note: Each thread will create its own session to avoid thread-safety issues


def call_api(request_id=None):
    """
    Simulates the exact flow from api_manager._run_cloud_ml_model:
    1. Embed text description
    2. Get predictions with vision + text embeddings

    Args:
        request_id: Optional request ID for tracking
    """
    # Add small delay to avoid rate limiting
    if REQUEST_DELAY > 0:
        time.sleep(REQUEST_DELAY)

    # Create a session per thread to avoid thread-safety issues
    session = requests.Session()

    try:
        # Step 1: Embed text description
        text_response = session.post(
            TEXT_EMBED_URL,
            json={"text": SAMPLE_TEXT_DESCRIPTION},
            timeout=60,
            verify=VERIFY_SSL,  # SSL verification using certificate file
        )

        # Debug: Print response details for first call
        if not hasattr(call_api, "debug_printed"):
            call_api.debug_printed = True
            print(f"\n=== Response Debug Info ===")
            print(f"Status Code: {text_response.status_code}")
            print(f"Response Headers: {dict(text_response.headers)}")
            print(f"Response Body: {text_response.text[:500]}")
            print(f"===========================\n")

        text_response.raise_for_status()
        text_embedding = text_response.json().get("embedding", [0.0] * EMBED_DIM)

        # Step 2: Get predictions
        prediction_response = session.post(
            PREDICTION_URL,
            json={
                "vision_embedding": SAMPLE_VISION_EMBEDDING,
                "text_embedding": text_embedding,
                "top_k": 5,
            },
            timeout=60,
            verify=VERIFY_SSL,  # SSL verification using certificate file
        )
        prediction_response.raise_for_status()
        result = prediction_response.json()

        # Print only once for sanity-check
        if not hasattr(call_api, "first_call"):
            call_api.first_call = True
            print("Sample response:\n", json.dumps(result, indent=2))

    except requests.exceptions.HTTPError as e:
        # Print detailed error information
        print(f"\n=== HTTP Error Details ===")
        print(f"Status Code: {e.response.status_code}")
        print(f"URL: {e.response.url}")
        print(f"Response Headers: {dict(e.response.headers)}")
        print(f"Response Body: {e.response.text}")
        print(f"==========================\n")
        raise
    except Exception as e:
        print(f"Error in API call: {e}")
        import traceback

        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    print(f"Starting load test with {NUM_WORKERS} workers, {NUM_REQUESTS} total requests...")
    print(f"Text Embed URL: {TEXT_EMBED_URL}")
    print(f"Prediction URL: {PREDICTION_URL}")
    print("=" * 70)

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        start_time = time.time()
        start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        print(f"\n🚀 Starting to send requests at: {start_timestamp}")
        print("=" * 70)

        for i in range(NUM_REQUESTS):
            # Add delay between submitting requests to avoid overwhelming the server
            if i > 0 and REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)
            future = executor.submit(call_api, request_id=i + 1)
            futures.append(future)
            if (i + 1) % 500 == 0:
                print(f"Submitted {i + 1} requests...")

        print("\nWaiting for all requests to complete...")
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 500 == 0:
                print(f"Completed {completed}/{NUM_REQUESTS} requests...")

    print(f"\nLoad test complete! All {completed} requests finished.")

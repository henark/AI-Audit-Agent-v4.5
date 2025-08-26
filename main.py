import os
import zai
from zai import ZaiClient
import logging
import logging.config
# Corrected import for the exception types
from zai.core import APIAuthenticationError, APIStatusError, APITimeoutError

def setup_debug_logging():
    """
    Configures logging to capture debug output from zai-sdk and httpx.
    This is the "debug mode" for the application.
    """
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'default',
            },
        },
        'loggers': {
            'zai': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': False,
            },
            'httpx': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': False,
            },
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)
    print("--- Debug logging enabled ---")

def main():
    """
    Initializes the ZaiClient and makes a simple API call.
    """
    setup_debug_logging()

    try:
        api_key = os.environ.get("ZAI_API_KEY")
        if not api_key or api_key == "your-api-key-placeholder":
            print("Error: ZAI_API_KEY environment variable not set or is a placeholder.")
            return

        client = ZaiClient(api_key=api_key)

        print("Making a test API call to Z.ai with model 'charglm-3'...")

        response = client.chat.completions.create(
            model="charglm-3",  # Changed model to one that is hopefully available
            messages=[
                {"role": "user", "content": "Hello, Z.ai! This is a test."}
            ]
        )

        print("\nAPI Response:")
        print(response.choices[0].message.content)

    # Corrected exception handling
    except APIAuthenticationError as err:
        print(f"\nAuthentication Error: {err}")
        print("Please ensure you have a valid API key.")
    except APIStatusError as err:
        print(f"\nAPI Status Error: {err}")
    except APITimeoutError as err:
        print(f"\nRequest Timeout: {err}")
    except Exception as err:
        print(f"\nAn unexpected error occurred: {err}")

if __name__ == "__main__":
    main()

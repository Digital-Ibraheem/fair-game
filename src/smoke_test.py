"""Smoke test to verify API-Football credentials work."""

import os
import sys
from dotenv import load_dotenv
import requests

def main():
    # Load environment variables
    load_dotenv()

    api_key = os.getenv('FOOTBALL_API_KEY')
    api_host = os.getenv('FOOTBALL_API_HOST', 'v3.football.api-sports.io')

    if not api_key or api_key == 'put_your_key_here':
        print("ERROR: FOOTBALL_API_KEY not set in .env file")
        print("Please add your API key to .env")
        sys.exit(1)

    # Test API connection
    url = f"https://{api_host}/status"
    headers = {
        'x-apisports-key': api_key
    }

    print(f"Testing API connection to {url}...")


    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()

        if data.get('errors'):
            print(f"ERROR: API returned errors: {data['errors']}")
            sys.exit(1)

        # Print account info
        account = data.get('response', {})
        print("\n✓ API Key Verified!")
        print(f"\nAccount Information:")
        print(f"  Name: {account.get('account', {}).get('firstname', 'N/A')} {account.get('account', {}).get('lastname', 'N/A')}")
        print(f"  Email: {account.get('account', {}).get('email', 'N/A')}")
        print(f"\nAPI Limits:")
        print(f"  Requests Limit (day): {account.get('requests', {}).get('limit_day', 'N/A')}")
        print(f"  Requests Used (day): {account.get('requests', {}).get('current', 'N/A')}")

        print("\n✓ Smoke test passed!")
        sys.exit(0)

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to connect to API: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
"""Thin root entrypoint for outbound Twilio dialing.

Usage:
  python call.py

Requires the Aiva worker (python agent.py) to be running first.
"""

import asyncio

from app.calling.dialer import main

if __name__ == "__main__":
    asyncio.run(main())

"""Thin root entrypoint for the LiveKit Agents worker.

Usage:
  python agent.py console
  python agent.py dev
  python agent.py start
"""

from app.agent.worker import run_worker

if __name__ == "__main__":
    run_worker()

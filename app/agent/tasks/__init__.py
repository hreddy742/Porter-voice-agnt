"""Conversation AgentTask subclasses for Aiva's call flow."""

from app.agent.tasks.booking import BookingTask
from app.agent.tasks.disclosure import DisclosureTask
from app.agent.tasks.exit import ExitTask
from app.agent.tasks.objection import ObjectionTask
from app.agent.tasks.opener import OpenerTask
from app.agent.tasks.pitch import PitchTask
from app.agent.tasks.qualifier import QualifierTask

__all__ = [
    "BookingTask",
    "DisclosureTask",
    "ExitTask",
    "ObjectionTask",
    "OpenerTask",
    "PitchTask",
    "QualifierTask",
]

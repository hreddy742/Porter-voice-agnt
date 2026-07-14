"""Pydantic models for lead intelligence data."""

from typing import Any

from pydantic import BaseModel


class Lead(BaseModel):
    """One callable lead returned by `get_next_lead`."""

    company_name: str
    city: str | None = None
    state: str | None = None
    industry: str | None = None
    naics_code: str | None = None
    website_domain: str | None = None
    phone: str | None = None
    lead_candidate_id: str
    tier: str | None = None
    current_score: float | int | None = None
    why_now_summary: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Dict form for AgentTask / supervisor code that uses `.get()`."""
        return self.model_dump()

import unittest

from app.agent.prompt import (
    SYSTEM_PROMPT,
    build_context_variables,
    build_system_prompt,
)


LEAD = {
    "company_name": "Example Co",
    "contact_name": "Sam",
    "city": "Austin",
    "state": "Texas",
    "industry": "staffing",
    "tier": "Warm",
}


class SingleSystemPromptTest(unittest.TestCase):
    def test_prompt_is_one_template_with_inline_knowledge(self):
        prompt = build_system_prompt(build_context_variables(LEAD))
        self.assertIn("Example Co", prompt)
        self.assertIn("Fees have a non-final range", prompt)
        self.assertNotIn("__COMPANY__", prompt)
        self.assertNotIn("PORTER_CAPITAL_KNOWLEDGE", SYSTEM_PROMPT)

    def test_prompt_contains_each_workflow_stage_once(self):
        prompt = build_system_prompt(build_context_variables(LEAD))
        for heading in ("Opener", "Pitch", "Qualifying", "Objection", "Booking", "Disclosure"):
            self.assertEqual(prompt.count(f"## {heading}"), 1)

    def test_prompt_owns_spoken_output_rules(self):
        prompt = build_system_prompt(build_context_variables(LEAD))
        for rule in ("Never output Markdown", "Do not read punctuation", "tool call must be silent"):
            self.assertIn(rule, prompt)

    def test_prompt_routes_explicit_hangups_to_livekit_tool(self):
        prompt = build_system_prompt(build_context_variables(LEAD))
        self.assertIn("explicitly asks to end or hang up", prompt)
        self.assertIn("user_ended", prompt)

    def test_context_variables_include_only_prompt_fields(self):
        lead = {
            **LEAD,
            "phone": "+15555550101",
            "current_score": 99,
            "why_now_summary": "Internal reason",
            "lead_candidate_id": "private-id",
            "website_domain": "example.com",
        }
        context = build_context_variables(lead)
        self.assertEqual(
            set(context),
            {"company", "contact", "city", "state", "industry", "tier"},
        )
        prompt = build_system_prompt(context)
        for private_value in (
            "+15555550101",
            "99",
            "Internal reason",
            "private-id",
            "example.com",
        ):
            self.assertNotIn(private_value, prompt)

    def test_missing_context_values_use_existing_fallbacks(self):
        context = build_context_variables({})
        self.assertEqual(
            context,
            {
                "company": "the company",
                "contact": "the contact",
                "city": "unknown",
                "state": "unknown",
                "industry": "unknown",
                "tier": "unknown",
            },
        )
        prompt = build_system_prompt(context)
        for placeholder in (
            "{company}",
            "{contact}",
            "{city}",
            "{state}",
            "{industry}",
            "{tier}",
        ):
            self.assertNotIn(placeholder, prompt)


if __name__ == "__main__":
    unittest.main()

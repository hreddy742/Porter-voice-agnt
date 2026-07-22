import asyncio

from dotenv import load_dotenv

from app.infrastructure.database import PostgresRepository


async def main() -> None:
    load_dotenv()
    repository = await PostgresRepository.connect()
    try:
        lead = await repository.get_next_lead()
    finally:
        await repository.close()
    if not lead:
        print("No leads available in database right now")
        return
    for label, key in (
        ("Company", "company_name"),
        ("City", "city"),
        ("State", "state"),
        ("Industry", "industry"),
        ("Phone", "phone"),
        ("Tier", "tier"),
        ("Score", "current_score"),
        ("Why Now", "why_now_summary"),
    ):
        print(f"{label}: {lead[key]}")


if __name__ == "__main__":
    asyncio.run(main())

import asyncio

from scripts.db_utils import connect


async def main() -> None:
    connection = await connect()
    try:
        checks = (
            ("Total companies", "SELECT COUNT(*) AS count FROM companies"),
            (
                "Lead candidates by sales status",
                "SELECT sales_status AS label, COUNT(*) AS count FROM lead_candidates GROUP BY sales_status",
            ),
            (
                "Lead candidates by tier",
                "SELECT tier AS label, COUNT(*) AS count FROM lead_candidates GROUP BY tier",
            ),
            (
                "Contactability statuses",
                "SELECT contactability_status AS label, COUNT(*) AS count FROM company_contactability GROUP BY contactability_status",
            ),
            (
                "Companies with phone numbers",
                "SELECT COUNT(*) AS count FROM company_contactability WHERE phone IS NOT NULL",
            ),
        )
        for title, query in checks:
            print(f"--- {title} ---")
            for row in await connection.fetch(query):
                print(dict(row))
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())

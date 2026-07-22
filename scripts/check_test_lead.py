import asyncio

from scripts.db_utils import connect


async def main() -> None:
    connection = await connect()
    try:
        rows = await connection.fetch(
            """
            SELECT c.canonical_name, lc.sales_status, cc.phone
            FROM lead_candidates lc
            JOIN companies c ON c.id = lc.company_id
            JOIN company_contactability cc ON cc.lead_candidate_id = lc.id
            WHERE cc.phone IS NOT NULL
            """
        )
        if not rows:
            print("No leads with phone numbers found")
        for row in rows:
            print(dict(row))
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())

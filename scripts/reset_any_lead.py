import asyncio

from scripts.db_utils import connect


async def main() -> None:
    connection = await connect()
    try:
        result = await connection.execute(
            """
            UPDATE lead_candidates
            SET sales_status = 'research', recontact_at = NULL, updated_at = now()
            WHERE company_id IN (
                SELECT company_id FROM company_contactability WHERE phone IS NOT NULL
            )
              AND sales_status != 'research'
            """
        )
        print(result)
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())

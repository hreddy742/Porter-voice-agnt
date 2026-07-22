import asyncio

from scripts.db_utils import connect


async def main() -> None:
    connection = await connect()
    try:
        result = await connection.execute(
            """
            UPDATE lead_candidates
            SET sales_status = 'research', recontact_at = NULL, updated_at = now()
            WHERE company_id = (
                SELECT id FROM companies
                WHERE canonical_name = 'Apex Staffing Solutions'
            )
            """
        )
        print(result)
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())

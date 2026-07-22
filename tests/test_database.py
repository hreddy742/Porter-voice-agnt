import unittest
from uuid import UUID

from app.infrastructure.database import PostgresRepository


class AsyncContext:
    def __init__(self, value=None):
        self.value = value or self

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_):
        return False


class FakeConnection:
    def __init__(self):
        self.executions = []
        self.transactions = 0

    def transaction(self):
        self.transactions += 1
        return AsyncContext()

    async def execute(self, query, *args):
        self.executions.append((query, args))
        return "UPDATE 1"


class FakePool:
    def __init__(self):
        self.connection = FakeConnection()
        self.closed = False
        self.fetchval_call = None
        self.fetchrow_call = None
        self.fetchrow_result = None

    def acquire(self):
        return AsyncContext(self.connection)

    async def fetchval(self, query, *args):
        self.fetchval_call = (query, args)
        return 17

    async def fetchrow(self, query, *args):
        self.fetchrow_call = (query, args)
        return self.fetchrow_result

    async def close(self):
        self.closed = True


class DatabaseTest(unittest.IsolatedAsyncioTestCase):
    async def test_lead_selection_allows_missing_phone(self):
        pool = FakePool()
        pool.fetchrow_result = {
            "lead_candidate_id": UUID("00000000-0000-0000-0000-000000000001"),
            "company_name": "Example Co",
            "phone": None,
        }
        repository = PostgresRepository(pool)

        lead = await repository.get_next_lead()

        query, _ = pool.fetchrow_call
        self.assertEqual(lead["phone"], None)
        self.assertIn("LEFT JOIN LATERAL", query)
        self.assertNotIn("cc.phone IS NOT NULL", query)
        self.assertEqual(
            lead["lead_candidate_id"],
            "00000000-0000-0000-0000-000000000001",
        )

    async def test_create_call_uses_asyncpg_placeholders(self):
        pool = FakePool()
        repository = PostgresRepository(pool)
        call_id = await repository.create_call(
            "00000000-0000-0000-0000-000000000001",
            "v2",
            "openai",
            "cartesia",
            "room",
        )
        query, args = pool.fetchval_call
        self.assertEqual(call_id, 17)
        self.assertIn("$1", query)
        self.assertNotIn("%s", query)
        self.assertIsInstance(args[0], UUID)
        self.assertEqual(args[2:4], ("openai", "cartesia"))

    async def test_call_and_lead_complete_in_one_transaction(self):
        pool = FakePool()
        repository = PostgresRepository(pool)
        await repository.complete_call(
            17,
            "00000000-0000-0000-0000-000000000001",
            "callback_booked",
            callback_details="Friday at ten",
        )
        self.assertEqual(pool.connection.transactions, 1)
        self.assertEqual(len(pool.connection.executions), 2)

    async def test_suppression_entries_share_one_transaction(self):
        pool = FakePool()
        repository = PostgresRepository(pool)
        await repository.add_to_suppression("Example Co", "example.com")
        self.assertEqual(pool.connection.transactions, 1)
        self.assertEqual(len(pool.connection.executions), 2)


if __name__ == "__main__":
    unittest.main()

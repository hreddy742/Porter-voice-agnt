import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from scripts import run_migrations as runner


ROOT = Path(__file__).parents[1]
MIGRATIONS = ROOT / "migrations"


class AsyncTransaction:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        self.connection.transactions_started += 1
        return self

    async def __aexit__(self, exc_type, *_):
        if exc_type is None:
            self.connection.transactions_committed += 1
        else:
            self.connection.transactions_rolled_back += 1
        return False


class FakeConnection:
    def __init__(self, *, history_exists=False, checksums=None, fail_sql=None):
        self.history_exists = history_exists
        self.checksums = dict(checksums or {})
        self.fail_sql = fail_sql
        self.executions = []
        self.transactions_started = 0
        self.transactions_committed = 0
        self.transactions_rolled_back = 0
        self.closed = False

    def transaction(self):
        return AsyncTransaction(self)

    async def fetchval(self, query, *args):
        if "to_regclass" in query:
            return "schema_migrations" if self.history_exists else None
        if "SELECT checksum" in query:
            return self.checksums.get(args[0])
        raise AssertionError(f"Unexpected query: {query}")

    async def execute(self, query, *args):
        self.executions.append((query, args))
        if query == self.fail_sql:
            raise RuntimeError("migration failed")
        if "CREATE TABLE IF NOT EXISTS schema_migrations" in query:
            self.history_exists = True
        if "INSERT INTO schema_migrations" in query:
            self.checksums[args[0]] = args[1]
        return "OK"

    async def close(self):
        self.closed = True


def migration(filename, sql):
    return runner.Migration(
        filename=filename,
        sql=sql,
        checksum=hashlib.sha256(sql.encode()).hexdigest(),
    )


class MigrationSchemaTest(unittest.TestCase):
    def test_every_required_table_has_an_ordered_sql_migration(self):
        files = sorted(MIGRATIONS.glob("*.sql"))
        self.assertEqual(
            [path.name for path in files],
            [
                "0001_database_prerequisites.sql",
                "0002_companies.sql",
                "0003_lead_candidates.sql",
                "0004_company_contactability.sql",
                "0005_suppression_list.sql",
                "0006_calls.sql",
            ],
        )
        sql = "\n".join(path.read_text(encoding="utf-8") for path in files)
        for table in (
            "schema_migrations",
            "companies",
            "lead_candidates",
            "company_contactability",
            "suppression_list",
            "calls",
        ):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", sql)

    def test_schema_contains_runtime_columns_constraints_and_indexes(self):
        sql = "\n".join(
            path.read_text(encoding="utf-8") for path in MIGRATIONS.glob("*.sql")
        )
        for column in (
            "canonical_name",
            "website_domain",
            "sector_excluded",
            "sales_status",
            "current_score",
            "why_now_summary",
            "contact_name",
            "recontact_at",
            "referral_details",
            "callback_details",
            "objection_text",
            "contactability_status",
            "match_type",
            "match_value",
            "final_call_result",
            "room_id",
        ):
            self.assertIn(column, sql)
        self.assertIn("REFERENCES companies(id)", sql)
        self.assertIn("REFERENCES lead_candidates(id)", sql)
        self.assertIn("UNIQUE (match_type, match_value)", sql)
        self.assertIn("DEFAULT gen_random_uuid()", sql)
        self.assertGreaterEqual(sql.count("CREATE INDEX IF NOT EXISTS"), 4)
        self.assertNotIn("call_turns", sql)
        self.assertNotIn("Apex Staffing Solutions", sql)

    def test_python_schema_migrations_are_removed(self):
        self.assertEqual(list((ROOT / "scripts").glob("migrate_*.py")), [])


class MigrationDiscoveryTest(unittest.TestCase):
    def test_discovers_in_filename_order_with_content_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0002_second.sql").write_text("SELECT 2;", encoding="utf-8")
            (root / "0001_first.sql").write_text("SELECT 1;", encoding="utf-8")

            migrations = runner.discover_migrations(root)

        self.assertEqual(
            [item.filename for item in migrations],
            ["0001_first.sql", "0002_second.sql"],
        )
        self.assertEqual(
            migrations[0].checksum,
            hashlib.sha256(b"SELECT 1;").hexdigest(),
        )

    def test_rejects_invalid_or_duplicate_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "migration.sql").write_text("SELECT 1;", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid migration filename"):
                runner.discover_migrations(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0001_first.sql").write_text("SELECT 1;", encoding="utf-8")
            (root / "0001_second.sql").write_text("SELECT 2;", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate migration sequence"):
                runner.discover_migrations(root)


class MigrationRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_applies_each_new_migration_in_its_own_transaction(self):
        migrations = [
            migration(
                "0001_database_prerequisites.sql",
                "CREATE TABLE IF NOT EXISTS schema_migrations (filename text);",
            ),
            migration("0002_companies.sql", "CREATE TABLE companies (id uuid);"),
        ]
        connection = FakeConnection()

        applied = await runner.run_migrations(connection, migrations)

        self.assertEqual(applied, [item.filename for item in migrations])
        self.assertEqual(connection.transactions_started, 2)
        self.assertEqual(connection.transactions_committed, 2)
        self.assertEqual(connection.transactions_rolled_back, 0)

    async def test_skips_applied_migration_with_matching_checksum(self):
        item = migration("0002_companies.sql", "SELECT 1;")
        connection = FakeConnection(
            history_exists=True,
            checksums={item.filename: item.checksum},
        )

        applied = await runner.run_migrations(connection, [item])

        self.assertEqual(applied, [])
        self.assertEqual(connection.transactions_started, 0)

    async def test_rejects_changed_applied_migration(self):
        item = migration("0002_companies.sql", "SELECT 1;")
        connection = FakeConnection(
            history_exists=True,
            checksums={item.filename: "0" * 64},
        )

        with self.assertRaisesRegex(RuntimeError, "checksum changed"):
            await runner.run_migrations(connection, [item])

        self.assertEqual(connection.transactions_started, 0)

    async def test_failure_rolls_back_and_stops_later_migrations(self):
        failing_sql = "BROKEN SQL"
        migrations = [
            migration("0002_failure.sql", failing_sql),
            migration("0003_never.sql", "SELECT 3;"),
        ]
        connection = FakeConnection(history_exists=True, fail_sql=failing_sql)

        with self.assertRaisesRegex(RuntimeError, "migration failed"):
            await runner.run_migrations(connection, migrations)

        self.assertEqual(connection.transactions_started, 1)
        self.assertEqual(connection.transactions_rolled_back, 1)
        self.assertNotIn("SELECT 3;", [query for query, _ in connection.executions])

    async def test_main_always_closes_connection(self):
        connection = FakeConnection(history_exists=True)
        with (
            patch.object(runner, "connect", AsyncMock(return_value=connection)),
            patch.object(runner, "discover_migrations", return_value=[]),
            patch.object(runner, "run_migrations", AsyncMock(side_effect=RuntimeError)),
        ):
            with self.assertRaises(RuntimeError):
                await runner.main()

        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()

import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host="localhost", port=5432,
    database="porter_leads",
    user="porter",
    password=os.getenv("DB_PASSWORD", "")
)
cursor = conn.cursor()

cursor.execute("""
    UPDATE lead_candidates
    SET sales_status = 'research', recontact_at = NULL, updated_at = now()
    WHERE company_id = (
        SELECT id FROM companies
        WHERE canonical_name = 'Apex Staffing Solutions'
    )
""")

conn.commit()
print(f"Rows updated: {cursor.rowcount}")
cursor.close()
conn.close()

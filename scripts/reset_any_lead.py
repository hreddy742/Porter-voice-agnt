import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    host='localhost', port=5432,
    database='porter_leads', user='porter',
    password=os.getenv('DB_PASSWORD','')
)
cursor = conn.cursor()

# Reset ANY lead that has a phone number back to research
cursor.execute("""
    UPDATE lead_candidates
    SET sales_status = 'research', recontact_at = NULL, updated_at = now()
    WHERE company_id IN (
        SELECT company_id FROM company_contactability
        WHERE phone IS NOT NULL
    )
    AND sales_status != 'research'
""")

conn.commit()
print(f'Rows reset: {cursor.rowcount}')
cursor.close()
conn.close()

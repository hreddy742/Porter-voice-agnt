import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="porter_leads",
    user="porter",
    password=os.getenv("DB_PASSWORD", "")
)
cursor = conn.cursor()

print('--- Total companies ---')
cursor.execute("SELECT COUNT(*) FROM companies")
print(cursor.fetchone()[0])

print('--- Lead candidates by sales_status ---')
cursor.execute("SELECT sales_status, COUNT(*) FROM lead_candidates GROUP BY sales_status")
for row in cursor.fetchall():
    print(' ', row[0], ':', row[1])

print('--- Lead candidates by tier ---')
cursor.execute("SELECT tier, COUNT(*) FROM lead_candidates GROUP BY tier")
for row in cursor.fetchall():
    print(' ', row[0], ':', row[1])

print('--- Contactability status counts ---')
cursor.execute("SELECT contactability_status, COUNT(*) FROM company_contactability GROUP BY contactability_status")
for row in cursor.fetchall():
    print(' ', row[0], ':', row[1])

print('--- Companies with phone numbers ---')
cursor.execute("SELECT COUNT(*) FROM company_contactability WHERE phone IS NOT NULL")
print(cursor.fetchone()[0])

cursor.close()
conn.close()

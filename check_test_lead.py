import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    host='localhost', port=5432,
    database='porter_leads', user='porter',
    password=os.getenv('DB_PASSWORD','')
)
cursor = conn.cursor()

cursor.execute("""
    SELECT c.canonical_name, lc.sales_status, cc.phone
    FROM lead_candidates lc
    JOIN companies c ON c.id = lc.company_id
    JOIN company_contactability cc ON cc.lead_candidate_id = lc.id
    WHERE cc.phone IS NOT NULL
""")

rows = cursor.fetchall()
if rows:
    for row in rows:
        print('Company:', row[0])
        print('Status: ', row[1])
        print('Phone:  ', row[2])
else:
    print('No leads with phone numbers found')

cursor.close()
conn.close()

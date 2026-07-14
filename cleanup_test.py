import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(host='localhost', port=5432,
    database='porter_leads', user='porter',
    password=os.getenv('DB_PASSWORD',''))
cursor = conn.cursor()
cursor.execute("""
    UPDATE company_contactability 
    SET phone = NULL 
    WHERE lead_candidate_id = '62c8587c-0286-4e41-a6e9-43557125ed87'
""")
conn.commit()
print('Test phone removed:', cursor.rowcount, 'rows')
cursor.close()
conn.close()

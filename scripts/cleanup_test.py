from app.db import get_connection

conn = get_connection()
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

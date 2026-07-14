from app.db import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    UPDATE lead_candidates
    SET sales_status = 'research', updated_at = now()
    WHERE company_id = (
        SELECT id FROM companies
        WHERE canonical_name = 'Apex Staffing Solutions'
    )
""")

conn.commit()
print(f"Rows updated: {cursor.rowcount}")
cursor.close()
conn.close()

from app.db import get_connection

conn = get_connection()
cursor = conn.cursor()

# Reset ANY lead that has a phone number back to research
cursor.execute("""
    UPDATE lead_candidates
    SET sales_status = 'research', updated_at = now()
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

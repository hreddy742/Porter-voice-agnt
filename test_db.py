from db import get_next_lead

lead = get_next_lead()

if lead:
    print('Lead found:')
    print('  Company :', lead['company_name'])
    print('  Location:', lead['city'], lead['state'])
    print('  Industry:', lead['industry'])
    print('  Phone   :', lead['phone'])
    print('  Tier    :', lead['tier'])
    print('  Score   :', lead['current_score'])
    print('  Why Now :', lead['why_now_summary'])
else:
    print('No leads available in database right now')

import pandas as pd
import json
import os

try:
    df = pd.read_excel(r'c:\Users\speec\OneDrive\Desktop\PoC_v2\data\WDPL_all_issues_with_P_processed.xlsx')
    
    # Filter for a specific issue if possible, or just take a sample
    # Let's check unique issues
    issues = df['issue'].unique()
    target_issue = issues[0] if len(issues) > 0 else None
    
    if target_issue:
        df_issue = df[df['issue'] == target_issue].copy()
    else:
        df_issue = df.head(100).copy()

    # Convert datetime to string
    df_issue['datetime_3h'] = df_issue['datetime_3h'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Select relevant columns
    cols = ['datetime_3h', 'channel', 'W', 'D', 'P', 'Lflag', 'Lnorm', 'C_cur', 'viewCount', 'comment_cnt']
    # Ensure columns exist
    cols = [c for c in cols if c in df_issue.columns]
    
    data = df_issue[cols].to_dict(orient='records')
    
    with open(r'c:\Users\speec\OneDrive\Desktop\PoC_v2\web\data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Extracted {len(data)} records for issue {target_issue}")

except Exception as e:
    print(f"Error: {e}")

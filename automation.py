from datetime import datetime
from plyer import notification
import pandas as pd

def check_followups(contacts_df):
    today = pd.Timestamp.today()
    contacts_df["Follow_Up_Date"] = pd.to_datetime(
        contacts_df["Follow_Up_Date"], errors="coerce"
    )

    overdue = contacts_df[
        contacts_df["Follow_Up_Date"] < today
    ]

    for _, row in overdue.iterrows():
        notification.notify(
            title="Job Follow-Up Reminder",
            message=f"Follow up with {row['Name']} ({row['Email']})",
            timeout=10,
        )

    return overdue

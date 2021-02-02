import pandas as pd
import email
from tqdm import tqdm

filepath = "../data/emails.csv"
emails = pd.read_csv(filepath)
print("Imported dataframe.")


# ============================================================
# Add the body of the messages as column
# ============================================================
def get_body(messages):
    column = []
    for message in messages:
        msg = email.message_from_string(message)
        msg = msg.get_payload()
        msg = msg.replace('\n', '')  # remove "\n" from the message
        column.append(msg)
    print("Added the body of the messages as a column.")
    return column


# ============================================================
# Add the subject of the messages as a column
# ============================================================
def get_field(field, messages):
    column = []
    for message in messages:
        msg = email.message_from_string(message)
        column.append(msg.get(field))
    print("Added the subject of the message as a column.")
    return column


# ============================================================
# Find all mails with distinct subjects
# ============================================================
def distinct_subject(emails):
    emails = emails.groupby("subject").filter(lambda x: len(x) == 1)
    print("Found", len(emails), "mails with distinct subject.")
    return emails


# ============================================================
# Find all replies
# ============================================================
def find_reply(emails):
    subjects_filtered = {}
    index = 0
    for i in emails["subject"]:
        i_lower = i.lower()
        if "re:" in i_lower:
            subjects_filtered[i] = emails.iloc[index, 2]
        index += 1
    print("Found", len(subjects_filtered), "replies.")
    return subjects_filtered


# ============================================================
# Find all pairs of messaage and reply
# ============================================================
def find_pairs(subjects_filtered, emails):
    subject = []
    resubject = []
    body = []
    reply = []

    for key, value in tqdm(subjects_filtered.items()):
        subj = key[4:]
        for i in range(len(emails)):
            if emails.iloc[i]["subject"] == subj:
                subject.append(emails.iloc[i]["subject"])
                resubject.append(key)
                body.append(emails.iloc[i]["body"])
                reply.append(value)
                break

    print("found all pairs of message and reply.")
    final = {"subject": subject, "re_subject": resubject, "body": body, "reply": reply}
    final = pd.DataFrame(data=final)
    return final


# ============================================================
# Save processed dataframe as a new csv file
# ============================================================
def save_csv(final):
    path = "../data/processed_emails.csv"
    final.to_csv(path)
    print("Preprocessing complete. New csv file saved in:", path)
    return


emails["body"] = get_body(emails["message"])
emails["subject"] = get_field("Subject", emails["message"])
emails = distinct_subject(emails)
subjects_filtered = find_reply(emails)
final = find_pairs(subjects_filtered, emails)
save_csv(final)

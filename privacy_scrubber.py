import pandas as pd
import spacy
import re
import os

# Load the NLP model (This model identifies Names, Places, and Dates)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Instructions for the user if they don't have the model
    print("Please run: python -m spacy download en_core_web_sm")

# Define how we want to label the redacted information
REDACTION_MAP = {
    "PERSON": "[NAME]",
    "GPE": "[LOCATION]",
    "LOC": "[LOCATION]",
    "ORG": "[ORGANIZATION]",
    "DATE": "[DATE]",
    "TIME": "[TIME]"
}

def scrub_privacy_data(text):
    """
    Removes personal info like names, emails, and phone numbers from text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step A: Use Patterns (Regex) to find standard formats
    text = re.sub(r'https?://\S+', '[URL]', text)              # Remove Links
    text = re.sub(r'(u/|@)\w+', '[USER]', text)                # Remove Usernames
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)                 # Remove Emails
    # Remove Phone Numbers
    text = re.sub(r'(\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)', '[PHONE]', text)

    # Step B: Use AI (SpaCy) to find Names and Locations
    doc = nlp(text)
    new_text = text
    
    # We work backwards to avoid messing up the text index
    for ent in reversed(doc.ents):
        if ent.label_ in REDACTION_MAP:
            replacement = REDACTION_MAP[ent.label_]
            new_text = new_text[:ent.start_char] + replacement + new_text[ent.end_char:]
            
    return new_text

def main():
    input_file = "reddit_data_filtered.csv"
    output_file = "reddit_data_anonymized.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print("Loading data and starting de-identification...")
    df = pd.read_csv(input_file)

    # Apply the scrubbing function to Title and Text
    print("Scrubbing Title column...")
    df['title'] = df['title'].apply(scrub_privacy_data)
    
    print("Scrubbing Body text column...")
    df['selftext'] = df['selftext'].fillna('').apply(scrub_privacy_data)

    # Save the cleaned data
    df.to_csv(output_file, index=False)
    print(f"Process complete! Anonymized data saved to {output_file}")

if __name__ == "__main__":
    main()
import praw
import prawcore
import pandas as pd
import datetime
import re
import emoji
from unidecode import unidecode

# Get your credentials at https://www.reddit.com/prefs/apps
CLIENT_ID = "YOUR_CLIENT_ID_HERE"
CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"
USER_AGENT = "ResearchDataCollector/1.0"

# List of subreddits to analyze( Take the ones you need )
SUBREDDITS = [
    "India", "LegalAdviceIndia", "AskIndia"
]

# Keywords related to the research 
# IMPORTANT: Select Keywords using Field Experts(takes a lot of time)
KEYWORDS = [
    "police", "cops"
]

# DATA COLLECTION FUNCTIONS 

def clean_text(text):
    """Cleans raw text for analysis: removes URLs, emojis, and special chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'u/\S+|@\S+', '', text)           # Remove mentions
    text = emoji.replace_emoji(text, replace='')     # Remove emojis
    text = re.sub(r'[^a-z\s]', '', text)             # Keep only letters/spaces
    text = re.sub(r'\s+', ' ', text).strip()         # Remove extra whitespace
    return text

def collect_reddit_data():
    """Connects to Reddit API and searches for relevant posts."""
    
    # Initialize Reddit connection
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

    collected_posts = []
    seen_ids = set()
    
    # Define date range (Jan 2022 to June 2025)
    start_ts = int(datetime.datetime(2022, 1, 1).timestamp())
    end_ts = int(datetime.datetime(2025, 6, 30).timestamp())

    print(f"Starting collection from {len(SUBREDDITS)} subreddits...")

    for sub_name in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub_name)
            # Combine keywords into a single search query (e.g., 'police OR cops')
            query = " OR ".join(KEYWORDS)
            
            for submission in subreddit.search(query, limit=None):
                if submission.id not in seen_ids:
                    if start_ts <= submission.created_utc <= end_ts:
                        collected_posts.append({
                            "id": submission.id,
                            "subreddit": sub_name,
                            "created_at": datetime.datetime.fromtimestamp(submission.created_utc),
                            "title": submission.title,
                            "text": submission.selftext,
                        })
                        seen_ids.add(submission.id)
            print(f"Finished r/{sub_name}")
            
        except Exception as e:
            print(f"Error in r/{sub_name}: {e}")

    return pd.DataFrame(collected_posts)

if __name__ == "__main__":
    df = collect_reddit_data()
    
    if not df.empty:
        print("\nCleaning data...")
        # Create full text and clean it
        df['raw_content'] = df['title'] + ' ' + df['text'].fillna('')
        df['cleaned_content'] = df['raw_content'].apply(clean_text)
        
        # Save to CSV
        output_file = "reddit_research_data.csv"
        df.to_csv(output_file, index=False)
        print(f"Success! {len(df)} posts saved to {output_file}")
    else:
        print("No posts found. Please check your credentials and keywords.")
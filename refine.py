import pandas as pd
import re
from tqdm.auto import tqdm

#  Change these to adapt the script for other topics
INPUT_FILE = "reddit_data_raw.csv"
OUTPUT_FILE = "reddit_data_filtered.csv"

# Posts must contain at least one of these words
INCLUSION_TERMS = [
    'police', 'cops'
]

# Posts containing these words will be removed (noise reduction)
EXCLUSION_TERMS = [
    'movie', 'song', 'band', 'game'
]

# LOGIC FUNCTIONS 

def is_post_relevant(row):
    """
    Checks if a post is actually about law enforcement.
    Returns True if relevant, False otherwise.
    """
    # Combine title and body text for a full search
    text = f"{row['title']} {row['selftext']}".lower()
    
    # Checks for noise first if it mentions a movie/game, discard it
    for word in EXCLUSION_TERMS:
        if word in text:
            return False
            
    # Check for relevance: does it mention a core police term?
    for word in INCLUSION_TERMS:
        if word in text:
            return True
            
    return False

# EXECUTION

def main():
    try:
        # Load the data
        print(f"Loading {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE)
        
        print("Filtering out irrelevant posts and noise...")
        tqdm.pandas()
        relevant_mask = df.progress_apply(is_post_relevant, axis=1)
        
        df_filtered = df[relevant_mask].copy()
        
        # Show results
        removed_count = len(df) - len(df_filtered)
        print(f"Success! Kept {len(df_filtered)} posts. Removed {removed_count} irrelevant posts.")
        
        # Save to CSV file
        df_filtered.to_csv(OUTPUT_FILE, index=False)
        print(f"Filtered data saved to: {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_FILE}' was not found.")

if __name__ == "__main__":
    main()
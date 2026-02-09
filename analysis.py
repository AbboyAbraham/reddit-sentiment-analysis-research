import pandas as pd
import numpy as np
import spacy
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#Initiate dataset and hardware selection
class RedditResearchAnalyzer:
    def __init__(self, input_file):
        self.df = pd.read_csv(input_file)
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        self.device = 0 if torch.cuda.is_available() else -1
        
    def run_sentiment_analysis(self):
        """Calculates general sentiment scores (-1 to +1 scale) using VADER."""
        analyzer = SentimentIntensityAnalyzer()
        
        self.df['sentiment_score'] = self.df['full_text'].apply(
            lambda x: analyzer.polarity_scores(str(x))['compound']
        )
        self.df['sentiment_label'] = pd.cut(
            self.df['sentiment_score'], 
            bins=[-1, -0.05, 0.05, 1], 
            labels=['Negative', 'Neutral', 'Positive']
        )
        return self.df

    def run_emotion_classification(self, batch_size=32):
        """Detects specific emotional archetypes using the DistilRoBERTa model."""
        classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base",
            device=self.device
        )
        
        texts = self.df['full_text'].str[:512].tolist()
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Emotion Analysis"):
            batch = texts[i:i+batch_size]
            results.extend(classifier(batch, truncation=True))
            
        self.df['dominant_emotion'] = [r['label'] for r in results]
        self.df['emotion_score'] = [r['score'] for r in results]
        return self.df

    def perform_topic_modeling(self, n_topics=5):
        """Groups posts into 5 thematic clusters using Latent Dirichlet Allocation (LDA)."""
        # Excludes non-analytical 'noise' words
        custom_sw = self.nlp.Defaults.stop_words | {'reddit', 'post', 'police', 'think', 'know'}
        
        def process_text(text):
            doc = self.nlp(str(text).lower())
            return " ".join([t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in custom_sw])

        tqdm.pandas(desc="Topic Modeling Prep")
        clean_text = self.df['full_text'].progress_apply(process_text)
        
        # Converts text into a word-frequency matrix for clustering
        vectorizer = CountVectorizer(max_df=0.9, min_df=10, ngram_range=(1, 2))
        dtm = vectorizer.fit_transform(clean_text)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.df['topic_cluster'] = lda.fit_transform(dtm).argmax(axis=1)
        
        return lda, vectorizer

    def save_results(self, output_file="research_results.csv"):
        self.df.to_csv(output_file, index=False)

if __name__ == "__main__":
    analyzer = RedditResearchAnalyzer("refined_reddit_data.csv")
    analyzer.run_sentiment_analysis()
    analyzer.run_emotion_classification()
    analyzer.perform_topic_modeling()
    analyzer.save_results()
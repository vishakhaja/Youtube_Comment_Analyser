import re
from googleapiclient.discovery import build
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import nltk
import sys
import io
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

# Redirect stdout to handle encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Extract video ID from a YouTube URL
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Initialize YouTube API client
api_key = "YOUR_API_KEY"  # Replace with your actual API key
youtube = build('youtube', 'v3', developerKey=api_key)

# Fetch comments from YouTube video
def fetch_comments(video_id):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
    except Exception as e:
        print(f"Error fetching comments: {e}")
    return comments

# Clean comments using pipeline
def clean_data_pipeline(comments):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    cleaned_comments = []
    for comment in comments:
        comment = comment.lower()
        comment = re.sub(r"http\S+", "", comment)  # Remove links
        comment = re.sub(r"@\w+", "", comment)  # Remove mentions
        comment = re.sub(r"[^\w\s]", "", comment)  # Remove special characters
        comment = re.sub(r"\s+", " ", comment).strip()  # Remove extra spaces

        tokens = word_tokenize(comment)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        cleaned_comment = " ".join(tokens)
        cleaned_comments.append(cleaned_comment)

    return cleaned_comments

def save_to_csv(comments, filename='cleaned_youtube_comments.csv'):
    """
    Save cleaned YouTube comments to a CSV file.
    
    Parameters:
    comments (list): List of cleaned YouTube comments.
    filename (str): The name of the CSV file to save. Default is 'cleaned_youtube_comments.csv'.
    """
    # Prepare data for saving
    data = {"Cleaned Comment": comments}
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Cleaned comments have been saved to {filename}")

# Get top N high-frequency keywords
def get_high_frequency_keywords(comments, top_n=5):
    all_words = " ".join(comments).split()
    word_freq = Counter(all_words)
    top_keywords = word_freq.most_common(top_n)
    return top_keywords

# Analyze sentiment
def analyze_sentiment(comments):
    sentiment_results = []
    for comment in comments:
        analysis = TextBlob(comment)
        sentiment = "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"
        sentiment_results.append(sentiment)
    return sentiment_results

# Plot sentiment distribution as bar chart
def plot_sentiment_bar_chart(sentiments):
    sentiment_counts = Counter(sentiments)
    df = pd.DataFrame(sentiment_counts.items(), columns=['Sentiment', 'Count'])
    fig = px.bar(df, x='Sentiment', y='Count', color='Sentiment', title="Sentiment Distribution",
                 color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'})
    fig.show()

# Keyword-based filtering for improvement suggestions
def keyword_filter_improvement_comments(comments):
    improvement_keywords = ["improve", "suggest", "recommend", "enhance", "better", "change", "develop"]
    
    improvement_comments = []
    for comment in comments:
        if any(keyword in comment for keyword in improvement_keywords):
            improvement_comments.append(comment)
    return improvement_comments

# Cluster positive comments using KMeans
def cluster_positive_comments(positive_comments, num_clusters=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(positive_comments)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)
    
    # Get cluster labels
    clusters = kmeans.labels_
    
    # Group comments by cluster
    clustered_comments = {}
    for i in range(num_clusters):
        clustered_comments[i] = []
    
    for idx, cluster in enumerate(clusters):
        clustered_comments[cluster].append(positive_comments[idx])
    
    return clustered_comments

def save_to_csv(comments, sentiments, filename='youtube_comments.csv'):
    """
    Save YouTube comments and their sentiment analysis results to a CSV file.
    
    Parameters:
    comments (list): List of YouTube comments.
    sentiments (list): List of sentiment labels corresponding to the comments.
    filename (str): The name of the CSV file to save. Default is 'youtube_comments.csv'.
    """
    # Prepare data for saving
    data = {
        "Comment": comments,
        "Sentiment": sentiments
    }

# Main script execution
url = "URL_OF_YOUTUBE"
video_id = extract_video_id(url)

if video_id:
    comments = fetch_comments(video_id)
    cleaned_comments = clean_data_pipeline(comments)

    # Display top 5 high-frequency keywords
    top_keywords = get_high_frequency_keywords(cleaned_comments)
    print("Top 5 High-Frequency Keywords:")
    for word, freq in top_keywords:
        print(f"{word}: {freq}")

    # Perform sentiment analysis
    sentiments = analyze_sentiment(cleaned_comments)
    plot_sentiment_bar_chart(sentiments)

    # Keyword-based filtering for improvement comments
    improvement_comments = keyword_filter_improvement_comments(cleaned_comments)
    print("\nComments Asking for Improvements:")
    for comment in improvement_comments:
        print(f" - {comment}")

    # Extract positive comments
    positive_comments = [comment for comment, sentiment in zip(cleaned_comments, sentiments) if sentiment == "Positive"]

    if positive_comments:
        # Cluster positive comments
        clustered_comments = cluster_positive_comments(positive_comments, num_clusters=3)
        
        # Print clusters of positive comments
        print("\nClusters of Positive Comments:")
        for cluster_num, comments_in_cluster in clustered_comments.items():
            print(f"\nCluster {cluster_num + 1}:")
            for comment in comments_in_cluster:
                print(f" - {comment}")
    else:
        print("No positive comments found.")
else:
    print("Invalid YouTube URL")

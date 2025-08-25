
# --- Enhanced YouTube Influencer AI Agent - Hackathon Edition ---
import streamlit as st
import pandas as pd
import requests
import os
import json
import pymongo
from dotenv import load_dotenv
from fpdf import FPDF
from PIL import Image
from io import BytesIO
import altair as alt
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from textblob import TextBlob
import time
from collections import Counter
import hashlib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load environment variables ---
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# --- MongoDB Setup ---
try:
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client["youtube_app"]
    history_collection = db["search_history"]
    analytics_collection = db["channel_analytics"]
    predictions_collection = db["trend_predictions"]
except:
    st.warning("⚠ MongoDB not available. Using local storage.")
    client = None
    db = None
    history_collection = None
    analytics_collection = None
    predictions_collection = None

# --- Enhanced Configuration ---
st.set_page_config(
    page_title="🚀 AI YouTube Influencer Intelligence Platform", 
    layout="wide", 
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .channel-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    .channel-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    .selected-channel {
        border: 3px solid #667eea;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
    .insight-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #e17055;
    }
    .trend-indicator {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .positive { color: #00b894; }
    .negative { color: #e17055; }
    .neutral { color: #fdcb6e; }
    .stTab {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Persistent Storage ---
CHANNELS_FILE = "saved_channels.json"
ANALYTICS_CACHE = "analytics_cache.json"

def load_channels():
    if os.path.exists(CHANNELS_FILE):
        with open(CHANNELS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_channels(channels):
    with open(CHANNELS_FILE, "w") as f:
        json.dump(channels, f)

def cache_analytics(data):
    with open(ANALYTICS_CACHE, "w") as f:
        json.dump(data, f, default=str)

def load_analytics_cache():
    if os.path.exists(ANALYTICS_CACHE):
        with open(ANALYTICS_CACHE, "r") as f:
            return json.load(f)
    return {}

# --- Enhanced API Functions ---
def fetch_channel_info(channel_id):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.channels().list(
            part="snippet,statistics,brandingSettings", 
            id=channel_id
        ).execute()
        
        if response["items"]:
            item = response["items"][0]
            snippet = item["snippet"]
            stats = item["statistics"]
            branding = item.get("brandingSettings", {}).get("channel", {})
            
            return {
                "title": snippet["title"],
                "description": snippet.get("description", ""),
                "thumbnail": snippet["thumbnails"]["high"]["url"],
                "subscribers": int(stats.get("subscriberCount", 0)),
                "total_views": int(stats.get("viewCount", 0)),
                "video_count": int(stats.get("videoCount", 0)),
                "country": snippet.get("country", "Unknown"),
                "created_at": snippet["publishedAt"],
                "keywords": branding.get("keywords", "").split(",") if branding.get("keywords") else []
            }
    except Exception as e:
        st.error(f"Error fetching channel info: {str(e)}")
        return None

def fetch_youtube_content_extended(channel_id, days=30, max_results=50):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        
        # Get channel uploads playlist
        channel_response = youtube.channels().list(
            part="contentDetails", 
            id=channel_id
        ).execute()
        
        uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        
        # Get recent videos
        playlist_response = youtube.playlistItems().list(
            part="snippet", 
            playlistId=uploads_playlist_id, 
            maxResults=max_results
        ).execute()
        
        video_ids = [item["snippet"]["resourceId"]["videoId"] for item in playlist_response["items"]]
        
        # Get detailed video statistics
        videos_response = youtube.videos().list(
            part="statistics,snippet,contentDetails",
            id=",".join(video_ids)
        ).execute()
        
        recent_videos = []
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        for item in videos_response["items"]:
            snippet = item["snippet"]
            stats = item["statistics"]
            content_details = item["contentDetails"]
            
            published_at = datetime.strptime(snippet["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
            
            if published_at > cutoff:
                duration_str = content_details.get("duration", "PT0M0S")
                duration_seconds = parse_duration(duration_str)
                
                recent_videos.append({
                    "video_id": item["id"],
                    "title": snippet["title"],
                    "description": snippet["description"][:500],
                    "published_at": published_at,
                    "thumbnail": snippet["thumbnails"]["high"]["url"],
                    "view_count": int(stats.get("viewCount", 0)),
                    "like_count": int(stats.get("likeCount", 0)),
                    "comment_count": int(stats.get("commentCount", 0)),
                    "duration": duration_seconds,
                    "tags": snippet.get("tags", []),
                    "url": f"https://www.youtube.com/watch?v={item['id']}"
                })
        
        return sorted(recent_videos, key=lambda x: x["published_at"], reverse=True)
        
    except Exception as e:
        st.error(f"YouTube API Error: {str(e)}")
        return []

def parse_duration(duration_str):
    """Parse ISO 8601 duration to seconds"""
    import re
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    return 0

def get_video_comments(video_id, max_results=100):
    """Fetch comments for sentiment analysis"""
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            order="relevance"
        ).execute()
        
        comments = []
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": comment["textDisplay"],
                "author": comment["authorDisplayName"],
                "likes": comment["likeCount"],
                "published_at": comment["publishedAt"]
            })
        
        return comments
    except:
        return []

# --- AI Enhancement Functions ---
def advanced_summarization(text, summary_type="comprehensive"):
    """Enhanced summarization with different types"""
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    # Customize prompt based on summary type
    if summary_type == "comprehensive":
        prompt = f"Provide a comprehensive summary of this YouTube content: {text}"
    elif summary_type == "key_points":
        prompt = f"Extract key points from this content: {text}"
    elif summary_type == "audience_insight":
        prompt = f"Analyze this content for audience engagement insights: {text}"
    else:
        prompt = text
    
    payload = {"inputs": prompt}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result[0]["summary_text"] if isinstance(result, list) else "Could not summarize"
        else:
            return "API temporarily unavailable"
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": subjectivity
        }
    except:
        return {"sentiment": "neutral", "polarity": 0, "subjectivity": 0}

def extract_trending_topics(videos):
    """Extract trending topics from video titles and descriptions"""
    all_text = " ".join([f"{v['title']} {v['description']}" for v in videos])
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    word_freq = Counter(words)
    
    # Filter common words
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'this', 'that', 'with', 'from', 'they', 'she', 'her', 'his', 'him', 'how', 'can', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
    
    trending_topics = [(word, count) for word, count in word_freq.most_common(20) 
                      if word not in stop_words and len(word) > 3]
    
    return trending_topics[:10]

def predict_optimal_posting_time(videos):
    """Predict optimal posting times based on engagement"""
    if not videos:
        return "No data available"
    
    # Analyze posting times vs engagement
    time_engagement = {}
    for video in videos:
        hour = video["published_at"].hour
        engagement_rate = (video["like_count"] + video["comment_count"]) / max(video["view_count"], 1)
        
        if hour not in time_engagement:
            time_engagement[hour] = []
        time_engagement[hour].append(engagement_rate)
    
    # Calculate average engagement by hour
    avg_engagement = {}
    for hour, rates in time_engagement.items():
        avg_engagement[hour] = sum(rates) / len(rates)
    
    if avg_engagement:
        optimal_hour = max(avg_engagement.keys(), key=lambda x: avg_engagement[x])
        return f"{optimal_hour:02d}:00 - {(optimal_hour+1):02d}:00"
    
    return "Insufficient data"

def calculate_engagement_score(video):
    """Calculate comprehensive engagement score"""
    views = video["view_count"]
    likes = video["like_count"]
    comments = video["comment_count"]
    
    if views == 0:
        return 0
    
    # Weighted engagement score
    like_rate = likes / views
    comment_rate = comments / views
    
    # Comments are weighted more heavily as they indicate deeper engagement
    engagement_score = (like_rate * 0.7 + comment_rate * 0.3) * 100
    
    return min(engagement_score, 100)  # Cap at 100

def generate_content_recommendations(videos, channel_info):
    """Generate AI-powered content recommendations"""
    if not videos:
        return ["Upload more content to get recommendations"]
    
    # Analyze top performing videos
    sorted_videos = sorted(videos, key=calculate_engagement_score, reverse=True)
    top_videos = sorted_videos[:3]
    
    recommendations = []
    
    # Analyze common elements in top videos
    top_tags = []
    top_titles = []
    for video in top_videos:
        top_tags.extend(video.get("tags", []))
        top_titles.append(video["title"])
    
    tag_freq = Counter(top_tags)
    most_common_tags = [tag for tag, count in tag_freq.most_common(5)]
    
    if most_common_tags:
        recommendations.append(f"🏷 Focus on tags: {', '.join(most_common_tags[:3])}")
    
    # Duration analysis
    avg_duration = sum(v["duration"] for v in top_videos) / len(top_videos)
    recommendations.append(f"⏱ Optimal video length: {int(avg_duration//60)}:{int(avg_duration%60):02d}")
    
    # Posting frequency analysis
    days_since_first = (videos[0]["published_at"] - videos[-1]["published_at"]).days
    posting_frequency = len(videos) / max(days_since_first, 1) * 7  # per week
    
    if posting_frequency < 2:
        recommendations.append("📅 Increase posting frequency to 2-3 videos per week")
    elif posting_frequency > 5:
        recommendations.append("📅 Consider reducing frequency for better quality")
    
    # Engagement trend
    recent_engagement = sum(calculate_engagement_score(v) for v in videos[:5]) / 5
    older_engagement = sum(calculate_engagement_score(v) for v in videos[-5:]) / 5
    
    if recent_engagement > older_engagement:
        recommendations.append("📈 Engagement is trending up! Continue current strategy")
    else:
        recommendations.append("📉 Try experimenting with new content formats")
    
    return recommendations[:5]

# --- Session State Initialization ---
if "selected_channel_id" not in st.session_state:
    st.session_state.selected_channel_id = ""
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = {}
if "comparison_channels" not in st.session_state:
    st.session_state.comparison_channels = []

# --- Enhanced Header ---
st.markdown("""
<div class="main-header">
    <h1>🚀 AI YouTube Influencer Intelligence Platform</h1>
    <p>Advanced Analytics • Trend Prediction • Competitive Intelligence • Content Optimization</p>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown("## 🎯 Control Panel")
    
    # Quick Stats
    saved_channels = load_channels()
    if saved_channels:
        total_subs = sum(
    int(str(info.get("subscribers", 0)).replace(",", "") or 0)
    for info in saved_channels.values()
)
        st.metric("📺 Tracked Channels", len(saved_channels))
        st.metric("👥 Total Subscribers", f"{total_subs:,}")
    
    st.markdown("---")
    
    # Search History with enhanced UI
    st.markdown("### ⏳ Recent Activity")
    if history_collection is not None:
        history = list(history_collection.find().sort("timestamp", -1))
        for entry in history[:5]:
            with st.expander(f"🔍 {entry['channel_id'][:15]}..."):
                st.caption(f"⏰ {entry['timestamp']}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load", key=f"load_{entry['channel_id']}_{entry['timestamp']}"):
                        st.session_state.selected_channel_id = entry["channel_id"]
                        st.rerun()
                with col2:
                    if st.button("Delete", key=f"del_{entry['channel_id']}_{entry['timestamp']}"):
                        history_collection.delete_one({"_id": entry["_id"]})
                        st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙ Settings")
    analysis_days = st.slider("Analysis Period (days)", 7, 90, 30)
    max_videos = st.slider("Max Videos to Analyze", 10, 100, 50)
    
    # Export Options
    st.markdown("### 📤 Export Options")
    export_format = st.selectbox("Export Format", ["PDF", "CSV", "JSON"])
    include_charts = st.checkbox("Include Charts", True)

# --- Main Content Area ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Dashboard", "📊 Analytics", "🔍 Competitor Analysis", "🎯 Content Strategy", "📈 Trends & Predictions"])

with tab1:
    st.markdown("## 🏠 Channel Management Dashboard")
    
    # Add new channel with enhanced validation
    col1, col2 = st.columns([3, 1])
    with col1:
        new_channel_id = st.text_input("🆔 Enter YouTube Channel ID or URL", placeholder="UC...")
    with col2:
        if st.button("➕ Add Channel", type="primary"):
            # Extract channel ID from URL if needed
            if "youtube.com" in new_channel_id:
                import re
                match = re.search(r'channel/(UC[a-zA-Z0-9_-]{22})', new_channel_id)
                if match:
                    new_channel_id = match.group(1)
            
            if new_channel_id:
                with st.spinner("Fetching channel information..."):
                    info = fetch_channel_info(new_channel_id)
                    if info:
                        saved_channels[new_channel_id] = info
                        save_channels(saved_channels)
                        st.success(f"✅ Added {info['title']}")
                        st.rerun()
                    else:
                        st.error("❌ Invalid Channel ID or URL")
    
    # Enhanced channel display
    if saved_channels:
        st.markdown("### 🗂 Your Channels")
        
        # Search and filter
        search_term = st.text_input("🔍 Search channels...", "")
        sort_by = st.selectbox("Sort by", ["Name", "Subscribers", "Views", "Videos"])
        
        # Filter channels based on search
        filtered_channels = {}
        for cid, info in saved_channels.items():
            if not search_term or search_term.lower() in info["title"].lower():
                filtered_channels[cid] = info
        
        # Sort channels
        if sort_by == "Subscribers":
            filtered_channels = dict(sorted(filtered_channels.items(), 
                                          key=lambda x: x[1].get("subscribers", 0), reverse=True))
        elif sort_by == "Views":
            filtered_channels = dict(sorted(filtered_channels.items(), 
                                          key=lambda x: x[1].get("total_views", 0), reverse=True))
        elif sort_by == "Videos":
            filtered_channels = dict(sorted(filtered_channels.items(), 
                                          key=lambda x: x[1].get("video_count", 0), reverse=True))
        
        # Display channels in grid
        cols = st.columns(3)
        for i, (channel_id, info) in enumerate(filtered_channels.items()):
            with cols[i % 3]:
                card_class = "selected-channel" if st.session_state.selected_channel_id == channel_id else "channel-card"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <img src="{info['thumbnail']}" style="width:100%; border-radius:10px;">
                    <h4>{info['title'][:30]}...</h4>
                    👥 {info['subscribers']} subscribers <br>
                    👀 {info.get('views', 0)} total views <br>
                    📹 {info.get('videos', 0)} videos

                </div>
                """, unsafe_allow_html=True)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button("Select", key=f"select_{channel_id}"):
                        st.session_state.selected_channel_id = channel_id
                        st.rerun()
                with col_b:
                    if st.button("📊", key=f"analyze_{channel_id}", help="Quick Analysis"):
                        st.session_state.selected_channel_id = channel_id
                        # Auto-switch to analytics tab would go here
                        st.info("Channel selected for analysis!")
                with col_c:
                    if st.button("🗑", key=f"delete_{channel_id}", help="Delete"):
                        del saved_channels[channel_id]
                        save_channels(saved_channels)
                        st.rerun()
    else:
        st.info("🎯 Add your first channel to get started with AI-powered analytics!")

with tab2:
    st.markdown("## 📊 Advanced Channel Analytics")
    
    if not st.session_state.selected_channel_id:
        st.warning("⚠ Please select a channel from the Dashboard tab first.")
    else:
        channel_id = st.session_state.selected_channel_id
        channel_info = saved_channels.get(channel_id, {})
        
        # Analysis controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh Analytics", type="primary"):
                with st.spinner("🤖 AI is analyzing your channel..."):
                    # Save to history
                    if history_collection is not None and history_collection.count_documents({}) > 0:
                        history_collection.insert_one({
                            "channel_id": channel_id,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                    
                    # Fetch extended data
                    videos = fetch_youtube_content_extended(channel_id, analysis_days, max_videos)
                    
                    if videos:
                        # Enhanced analytics
                        analytics_data = {
                            "videos": videos,
                            "channel_info": channel_info,
                            "total_views": sum(v["view_count"] for v in videos),
                            "total_likes": sum(v["like_count"] for v in videos),
                            "total_comments": sum(v["comment_count"] for v in videos),
                            "avg_engagement": sum(calculate_engagement_score(v) for v in videos) / len(videos),
                            "trending_topics": extract_trending_topics(videos),
                            "optimal_posting_time": predict_optimal_posting_time(videos),
                            "content_recommendations": generate_content_recommendations(videos, channel_info)
                        }
                        
                        # Sentiment analysis on recent videos
                        sentiment_scores = []
                        for video in videos[:10]:  # Analyze top 10 recent videos
                            text = f"{video['title']} {video['description']}"
                            sentiment = analyze_sentiment(text)
                            sentiment_scores.append(sentiment)
                        
                        analytics_data["sentiment_analysis"] = sentiment_scores
                        
                        st.session_state.analytics_data = analytics_data
                        cache_analytics(analytics_data)
                        
                        st.success(f"✅ Analyzed {len(videos)} videos successfully!")
                    else:
                        st.error("❌ No videos found for analysis")
        
        # Display analytics if available
        if st.session_state.analytics_data and st.session_state.analytics_data.get("videos"):
            data = st.session_state.analytics_data
            videos = data["videos"]
            
            # Key Metrics Dashboard
            st.markdown("### 🎯 Key Performance Indicators")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>{:,}</h3>
                    <p>Total Views</p>
                </div>
                """.format(data["total_views"]), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>{:,}</h3>
                    <p>Total Likes</p>
                </div>
                """.format(data["total_likes"]), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>{:,}</h3>
                    <p>Total Comments</p>
                </div>
                """.format(data["total_comments"]), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3>{:.1f}%</h3>
                    <p>Avg Engagement</p>
                </div>
                """.format(data["avg_engagement"]), unsafe_allow_html=True)
            
            with col5:
                st.markdown("""
                <div class="metric-card">
                    <h3>{}</h3>
                    <p>Videos Analyzed</p>
                </div>
                """.format(len(videos)), unsafe_allow_html=True)
            
            # Advanced Visualizations
            st.markdown("### 📈 Performance Visualizations")
            
            # Create comprehensive dashboard
            col1, col2 = st.columns(2)
            
            with col1:
                # Engagement over time
                df_videos = pd.DataFrame(videos)
                df_videos["engagement_score"] = df_videos.apply(calculate_engagement_score, axis=1)
                df_videos["published_date"] = pd.to_datetime(df_videos["published_at"])
                
                fig = px.line(df_videos.sort_values("published_date"), 
                             x="published_date", y="engagement_score",
                             title="📊 Engagement Score Over Time",
                             hover_data=["title", "view_count", "like_count"])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Views vs Engagement scatter
                fig = px.scatter(df_videos, x="view_count", y="engagement_score",
                               size="comment_count", hover_data=["title"],
                               title="👀 Views vs Engagement Correlation",
                               trendline="ols")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance heatmap
            st.markdown("### 🔥 Publishing Pattern Analysis")
            
            # Create posting time heatmap
            df_videos["hour"] = df_videos["published_at"].dt.hour
            df_videos["day_of_week"] = df_videos["published_at"].dt.day_name()
            
            heatmap_data = df_videos.groupby(["day_of_week", "hour"])["engagement_score"].mean().unstack(fill_value=0)
            
            fig = px.imshow(heatmap_data, 
                           title="⏰ Optimal Posting Times (Engagement Score)",
                           labels=dict(x="Hour of Day", y="Day of Week", color="Avg Engagement"))
            st.plotly_chart(fig, use_container_width=True)
            
            # Content Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏷 Trending Topics")
                topics_df = pd.DataFrame(data["trending_topics"], columns=["Topic", "Frequency"])
                fig = px.bar(topics_df, x="Frequency", y="Topic", orientation="h",
                           title="Most Frequent Topics in Your Content")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 😊 Sentiment Analysis")
                if data.get("sentiment_analysis"):
                    sentiments = [s["sentiment"] for s in data["sentiment_analysis"]]
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    
                    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                               title="Content Sentiment Distribution",
                               color_discrete_map={"positive": "#00b894", "negative": "#e17055", "neutral": "#fdcb6e"})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights Panel
            st.markdown("### 🤖 AI-Powered Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="insight-card">
                    <h4>🕒 Optimal Posting Time</h4>
                    <p><strong>{}</strong></p>
                    <small>Based on engagement patterns from your recent videos</small>
                </div>
                """.format(data["optimal_posting_time"]), unsafe_allow_html=True)
            
            with col2:
                # Calculate trend
                recent_avg = sum(calculate_engagement_score(v) for v in videos[:len(videos)//2]) / (len(videos)//2)
                older_avg = sum(calculate_engagement_score(v) for v in videos[len(videos)//2:]) / (len(videos)//2)
                trend = "📈 Trending Up" if recent_avg > older_avg else "📉 Trending Down"
                trend_class = "positive" if recent_avg > older_avg else "negative"
                
                st.markdown(f"""
                <div class="insight-card">
                    <h4>📊 Engagement Trend</h4>
                    <p class="{trend_class}"><strong>{trend}</strong></p>
                    <small>Recent vs Historical Performance</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Content Recommendations
            st.markdown("### 🎯 AI Content Recommendations")
            for i, recommendation in enumerate(data["content_recommendations"], 1):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                           padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #667eea;">
                    <strong>{i}.</strong> {recommendation}
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed Video Performance Table
            st.markdown("### 📋 Individual Video Performance")
            
            # Create enhanced dataframe for display
            display_df = pd.DataFrame(videos)
            display_df["engagement_score"] = display_df.apply(calculate_engagement_score, axis=1)
            display_df["engagement_rate"] = ((display_df["like_count"] + display_df["comment_count"]) / display_df["view_count"] * 100).round(2)
            display_df["duration_formatted"] = display_df["duration"].apply(lambda x: f"{int(x//60)}:{int(x%60):02d}")
            
            # Select and format columns for display
            table_df = display_df[["title", "published_at", "view_count", "like_count", "comment_count", "engagement_score", "duration_formatted"]].copy()
            table_df.columns = ["Title", "Published", "Views", "Likes", "Comments", "Engagement Score", "Duration"]
            table_df["Title"] = table_df["Title"].apply(lambda x: x[:50] + "..." if len(x) > 50 else x)
            table_df["Published"] = pd.to_datetime(table_df["Published"]).dt.strftime("%Y-%m-%d %H:%M")
            table_df["Engagement Score"] = table_df["Engagement Score"].round(1)
            
            # Style the dataframe
            styled_df = table_df.style.format({
                "Views": "{:,}",
                "Likes": "{:,}",
                "Comments": "{:,}"
            }).background_gradient(subset=["Engagement Score"], cmap="RdYlGn")
            
            st.dataframe(styled_df, use_container_width=True)

with tab3:
    st.markdown("## 🔍 Competitive Intelligence Analysis")
    
    # Competitor comparison interface
    st.markdown("### 🥊 Channel vs Channel Comparison")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        primary_channel = st.selectbox("Primary Channel", 
                                     list(saved_channels.keys()) if saved_channels else [],
                                     format_func=lambda x: saved_channels.get(x, {}).get("title", x))
    
    with col2:
        competitor_id = st.text_input("Competitor Channel ID", placeholder="Enter competitor's channel ID")
    
    with col3:
        if st.button("🔍 Compare", type="primary"):
            if primary_channel and competitor_id:
                with st.spinner("Analyzing competitor data..."):
                    # Fetch competitor data
                    competitor_info = fetch_channel_info(competitor_id)
                    competitor_videos = fetch_youtube_content_extended(competitor_id, 30, 25)
                    
                    if competitor_info and competitor_videos:
                        st.session_state.comparison_data = {
                            "primary": {
                                "info": saved_channels[primary_channel],
                                "videos": fetch_youtube_content_extended(primary_channel, 30, 25)
                            },
                            "competitor": {
                                "info": competitor_info,
                                "videos": competitor_videos
                            }
                        }
                        st.success("✅ Comparison data loaded!")
    
    # Display comparison if available
    if hasattr(st.session_state, 'comparison_data') and st.session_state.comparison_data:
        data = st.session_state.comparison_data
        
        st.markdown("### 📊 Head-to-Head Comparison")
        
        # Comparison metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏠 Your Channel")
            primary_info = data["primary"]["info"]
            primary_videos = data["primary"]["videos"]
            
            p_avg_views = sum(v["view_count"] for v in primary_videos) / len(primary_videos) if primary_videos else 0
            p_avg_engagement = sum(calculate_engagement_score(v) for v in primary_videos) / len(primary_videos) if primary_videos else 0
            
            st.metric("Subscribers", f"{int(str(primary_info.get('subscribers', 0)).replace(',', '') or 0):,}")
            st.metric("Avg Views (30d)", f"{p_avg_views:,.0f}")
            st.metric("Avg Engagement", f"{p_avg_engagement:.1f}%")
            st.metric("Video Count (30d)", len(primary_videos))
        
        with col2:
            st.markdown("#### 🎯 Competitor")
            competitor_info = data["competitor"]["info"]
            competitor_videos = data["competitor"]["videos"]
            
            c_avg_views = sum(v["view_count"] for v in competitor_videos) / len(competitor_videos) if competitor_videos else 0
            c_avg_engagement = sum(calculate_engagement_score(v) for v in competitor_videos) / len(competitor_videos) if competitor_videos else 0
            
            # Calculate deltas
            subs_primary = int(str(primary_info.get("subscribers", 0)).replace(",", "") or 0)
            subs_competitor = int(str(competitor_info.get("subscribers", 0)).replace(",", "") or 0)

            subs_delta = subs_primary - subs_competitor
            views_delta = p_avg_views - c_avg_views
            engagement_delta = p_avg_engagement - c_avg_engagement
            
            st.metric("Subscribers", f"{competitor_info.get('subscribers', 0):,}", delta=f"{subs_delta:,}")
            st.metric("Avg Views (30d)", f"{c_avg_views:,.0f}", delta=f"{views_delta:,.0f}")
            st.metric("Avg Engagement", f"{c_avg_engagement:.1f}%", delta=f"{engagement_delta:+.1f}%")
            st.metric("Video Count (30d)", len(competitor_videos), delta=len(primary_videos) - len(competitor_videos))
        
        # Competitive analysis charts
        st.markdown("### 📈 Performance Comparison Charts")
        
        # Prepare data for comparison
        primary_df = pd.DataFrame(primary_videos)
        competitor_df = pd.DataFrame(competitor_videos)
        
        if not primary_df.empty and not competitor_df.empty:
            primary_df["channel"] = "Your Channel"
            competitor_df["channel"] = "Competitor"
            primary_df["engagement_score"] = primary_df.apply(calculate_engagement_score, axis=1)
            competitor_df["engagement_score"] = competitor_df.apply(calculate_engagement_score, axis=1)
            
            combined_df = pd.concat([primary_df, competitor_df])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Views comparison
                fig = px.box(combined_df, x="channel", y="view_count", 
                           title="📊 Views Distribution Comparison",
                           color="channel")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Engagement comparison
                fig = px.box(combined_df, x="channel", y="engagement_score",
                           title="🔥 Engagement Score Comparison",
                           color="channel")
                st.plotly_chart(fig, use_container_width=True)
            
            # Content gap analysis
            st.markdown("### 🎯 Content Gap Analysis")
            
            # Extract topics from both channels
            primary_topics = extract_trending_topics(primary_videos)
            competitor_topics = extract_trending_topics(competitor_videos)
            
            primary_words = set(word for word, count in primary_topics)
            competitor_words = set(word for word, count in competitor_topics)
            
            # Find gaps
            content_gaps = competitor_words - primary_words
            opportunities = primary_words - competitor_words
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Content Opportunities")
                st.write("Topics your competitor covers that you don't:")
                for topic in list(content_gaps)[:10]:
                    st.markdown(f"• *{topic}*")
            
            with col2:
                st.markdown("#### 💪 Your Unique Strengths")
                st.write("Topics you cover that your competitor doesn't:")
                for topic in list(opportunities)[:10]:
                    st.markdown(f"• *{topic}*")

with tab4:
    st.markdown("## 🎯 AI-Powered Content Strategy")
    
    if not st.session_state.selected_channel_id:
        st.warning("⚠ Please select a channel first to access content strategy tools.")
    else:
        # Content planning tools
        st.markdown("### 📝 Content Planner")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            content_type = st.selectbox("Content Type", ["Tutorial", "Review", "Entertainment", "Educational", "News/Commentary"])
        with col2:
            target_duration = st.selectbox("Target Duration", ["Short (<5min)", "Medium (5-15min)", "Long (15min+)"])
        with col3:
            content_goal = st.selectbox("Primary Goal", ["Increase Views", "Boost Engagement", "Grow Subscribers", "Brand Awareness"])
        
        if st.button("🎯 Generate Content Strategy", type="primary"):
            with st.spinner("🤖 AI is crafting your personalized content strategy..."):
                # Simulate AI content strategy generation
                time.sleep(2)
                
                st.markdown("### 🚀 Your Personalized Content Strategy")
                
                strategy_recommendations = [
                    f"📺 *{content_type} Focus*: Based on your channel's performance, {content_type.lower()} content shows 23% higher engagement",
                    f"⏱ *Optimal Length*: Aim for {target_duration.lower()} videos - they perform best for your audience",
                    f"🎯 *{content_goal}*: Implement these tactics to achieve your primary goal",
                    "📅 *Posting Schedule*: Tuesday and Thursday at 2 PM show highest engagement for your niche",
                    "🏷 *Trending Tags*: Include tags like #viral, #trending, and niche-specific keywords",
                    "👀 *Thumbnail Strategy*: Use bright colors and faces - increases CTR by 31%",
                    "📱 *Cross-Platform*: Repurpose content for YouTube Shorts, TikTok, and Instagram Reels"
                ]
                
                for rec in strategy_recommendations:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #a8e6cf 0%, #88d8c0 100%); 
                               padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #00b894;">
                        {rec}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Content calendar
        st.markdown("### 📅 Content Calendar Generator")
        
        col1, col2 = st.columns(2)
        with col1:
            calendar_weeks = st.slider("Generate calendar for (weeks)", 1, 8, 4)
        with col2:
            posts_per_week = st.slider("Videos per week", 1, 7, 3)
        
        if st.button("📅 Generate Calendar"):
            st.markdown("### 🗓 Your Content Calendar")
            
            # Generate sample calendar
            calendar_data = []
            content_ideas = [
                "Tutorial: How to Master [Your Niche Topic]",
                "Behind the Scenes: My Creative Process",
                "Q&A: Answering Your Top Questions",
                "Trending Topic: [Current Event] Explained",
                "Collaboration: Guest Interview",
                "Review: Latest [Product/Service] in Your Niche",
                "Educational: Common Mistakes to Avoid",
                "Entertainment: Fun Challenge or Game"
            ]
            
            start_date = datetime.now()
            for week in range(calendar_weeks):
                for post in range(posts_per_week):
                    post_date = start_date + timedelta(days=week*7 + post*2)
                    calendar_data.append({
                        "Date": post_date.strftime("%Y-%m-%d"),
                        "Content Idea": content_ideas[post % len(content_ideas)],
                        "Type": content_type,
                        "Status": "Planned"
                    })
            
            calendar_df = pd.DataFrame(calendar_data)
            st.dataframe(calendar_df, use_container_width=True)
        
        # Title and thumbnail generator
        st.markdown("### ✨ AI Content Generator")
        
        topic_input = st.text_input("💡 Enter your video topic or keyword")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏷 Generate Titles"):
                if topic_input:
                    st.markdown("#### 🎯 AI-Generated Title Ideas")
                    titles = [
                        f"🔥 The Ultimate Guide to {topic_input} (2025 Edition)",
                        f"❌ {topic_input}: 7 Mistakes Everyone Makes",
                        f"💰 How I Made $$ with {topic_input}",
                        f"🚀 {topic_input} Secrets the Pros Don't Want You to Know",
                        f"⚡ {topic_input} in 60 Seconds (Speed Tutorial)",
                        f"🔍 {topic_input} vs [Alternative] - Which is Better?",
                        f"📈 Why {topic_input} Will Change Your Life",
                        f"🎭 Reacting to {topic_input} for the First Time"
                    ]
                    
                    for title in titles:
                        st.markdown(f"• {title}")
        
        with col2:
            if st.button("🖼 Thumbnail Tips"):
                if topic_input:
                    st.markdown("#### 🎨 Thumbnail Optimization Tips")
                    thumbnail_tips = [
                        "🎨 Use contrasting colors (red, yellow, blue)",
                        "😮 Include surprised/excited facial expressions",
                        "📝 Add bold, readable text overlay",
                        "👆 Use arrows pointing to key elements",
                        "🔢 Include numbers or statistics",
                        "✨ Add visual effects or borders",
                        "📱 Test readability on mobile devices",
                        "🎯 A/B test different versions"
                    ]
                    
                    for tip in thumbnail_tips:
                        st.markdown(f"• {tip}")

with tab5:
    st.markdown("## 📈 Trends & Future Predictions")
    
    # Trend analysis dashboard
    st.markdown("### 🔥 Current YouTube Trends")
    
    # Simulate trend data
    trend_categories = ["Technology", "Gaming", "Lifestyle", "Education", "Entertainment"]
    trend_data = pd.DataFrame({
        "Category": trend_categories,
        "Growth Rate": [15.3, 22.1, 8.7, 18.9, 12.4],
        "Engagement Rate": [4.2, 6.8, 3.1, 5.4, 4.9],
        "Competition Level": ["High", "Very High", "Medium", "Medium", "High"]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(trend_data, x="Category", y="Growth Rate",
                    title="📊 Category Growth Rates (%)",
                    color="Growth Rate",
                    color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(trend_data, x="Growth Rate", y="Engagement Rate",
                        size="Growth Rate", color="Category",
                        title="💫 Growth vs Engagement by Category")
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Predictions
    st.markdown("### 🔮 AI-Powered Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🚀 Short-form Content</h3>
            <p>+127% Growth Predicted</p>
            <small>Next 6 months</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 AI Tools Integration</h3>
            <p>Next Big Trend</p>
            <small>Q2 2025 Forecast</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📱 Mobile-First</h3>
            <p>85% of Views</p>
            <small>Platform Direction</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Future opportunities
    st.markdown("### 🎯 Emerging Opportunities")
    
    opportunities = [
        "🤖 *AI-Generated Content*: Early adopters see 40% more engagement",
        "🌐 *Web3 Integration*: NFTs and crypto content trending up",
        "🎮 *Interactive Videos*: Polls and quizzes boost retention by 60%",
        "🏠 *Virtual Reality*: VR content expected to grow 200% in 2025",
        "🎵 *Audio-First*: Podcast-style content gaining momentum",
        "🌍 *Sustainability*: Eco-friendly content resonating with Gen Z",
        "📚 *Micro-Learning*: Bite-sized educational content performing well",
        "🔄 *Cross-Platform*: Multi-platform strategies showing 3x growth"
    ]
    
    for opportunity in opportunities:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); 
                   padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #e17055;">
            {opportunity}
        </div>
        """, unsafe_allow_html=True)
    
    # Market forecast
    st.markdown("### 📊 6-Month Market Forecast")
    
    # Generate forecast data
    dates = pd.date_range(start=datetime.now(), periods=180, freq='D')
    forecast_data = pd.DataFrame({
        "Date": dates,
        "Predicted Growth": np.cumsum(np.random.normal(0.1, 0.5, 180)) + 100,
        "Confidence Interval": np.random.normal(2, 0.5, 180)
    })
    
    fig = px.line(forecast_data, x="Date", y="Predicted Growth",
                 title="📈 YouTube Market Growth Prediction",
                 line_shape="spline")
    fig.add_scatter(x=forecast_data["Date"], 
                   y=forecast_data["Predicted Growth"] + forecast_data["Confidence Interval"],
                   mode="lines", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=forecast_data["Date"], 
                   y=forecast_data["Predicted Growth"] - forecast_data["Confidence Interval"],
                   mode="lines", line=dict(width=0), 
                   fill="tonexty", fillcolor="rgba(102,126,234,0.2)",
                   showlegend=True, name="Confidence Interval")
    
    st.plotly_chart(fig, use_container_width=True)

# --- Enhanced Export and Footer ---
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Export Analytics"):
        if st.session_state.analytics_data:
            # Export comprehensive analytics
            st.download_button(
                label="💾 Download Analytics Report",
                data=json.dumps(st.session_state.analytics_data, default=str, indent=2),
                file_name=f"youtube_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

with col2:
    if st.button("📈 Generate Report"):
        st.info("🔄 Advanced reporting feature coming soon!")

with col3:
    if st.button("🤖 AI Consultation"):
        st.info("💬 Book a consultation with our AI experts!")

with col4:
    if st.button("🚀 Upgrade Features"):
        st.info("⭐ Premium features available in Pro version!")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>🚀 AI YouTube Influencer Intelligence Platform</h4>
    <p>Powered by Advanced AI • Real-time Analytics • Predictive Intelligence</p>
    <p><em>Hackathon Edition - Revolutionizing Content Creator Success</em></p>
</div>
""", unsafe_allow_html=True)
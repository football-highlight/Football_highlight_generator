"""
Streamlit web interface for Football Highlights Generator
"""

import streamlit as st
import requests
import json
import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import logging
import tempfile
import base64

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Football Highlights Generator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #424242;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .highlight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .event-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #1E88E5;
        color: white;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    .progress-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        transition: width 0.5s ease-in-out;
    }
    .video-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration - Make sure this matches your FastAPI server
API_BASE_URL = "http://localhost:8000/api"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
        else:
            logger.warning(f"API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to API server")
        return False
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return False

def upload_video(file, metadata=None):
    """Upload video to API"""
    try:
        # Prepare the files dict
        files = {"file": (file.name, file.getvalue(), file.type)}
        
        # Prepare data dict
        data = {}
        if metadata:
            # Ensure all metadata values are JSON serializable
            serializable_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    serializable_metadata[key] = value.isoformat()
                else:
                    serializable_metadata[key] = str(value)
            data["metadata"] = json.dumps(serializable_metadata)
        
        logger.info(f"Uploading {file.name} to {API_BASE_URL}/upload")
        
        # Make the request
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=30
        )
        
        logger.info(f"Upload response status: {response.status_code}")
        
        if response.status_code == 201:
            result = response.json()
            logger.info(f"Upload successful: {result}")
            return result
        else:
            logger.error(f"Upload failed: {response.text}")
            return {"status": "error", "message": f"Upload failed: {response.text}"}
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {"status": "error", "message": str(e)}

def start_processing(filename=None, video_path=None, generate_highlights=True):
    """Start video processing"""
    try:
        logger.info(f"Starting processing with filename={filename}, video_path={video_path}")
        
        # Prepare request data
        data = {
            "generate_highlights": generate_highlights
        }
        
        if filename:
            data["filename"] = filename
            logger.info(f"Using filename: {filename}")
        elif video_path:
            data["video_path"] = str(video_path)
            logger.info(f"Using video_path: {video_path}")
        else:
            return {"status": "error", "message": "No filename or video_path provided"}
        
        # Make the request
        logger.info(f"Sending request to {API_BASE_URL}/process with data: {data}")
        response = requests.post(
            f"{API_BASE_URL}/process",
            json=data,
            timeout=10
        )
        
        logger.info(f"Processing response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Processing started: {result}")
            return result
        else:
            error_msg = f"API returned status {response.status_code}: {response.text}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
            
    except Exception as e:
        error_msg = f"Processing request failed: {e}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

def get_job_status(job_id):
    """Get job status"""
    try:
        response = requests.get(f"{API_BASE_URL}/status/{job_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get job status: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return None

def list_videos():
    """List available videos"""
    try:
        response = requests.get(f"{API_BASE_URL}/videos", timeout=5)
        if response.status_code == 200:
            result = response.json()
            return result.get("videos", [])
        else:
            logger.warning(f"Failed to list videos: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        return []

def list_highlights():
    """List generated highlights"""
    try:
        response = requests.get(f"{API_BASE_URL}/highlights", timeout=5)
        if response.status_code == 200:
            result = response.json()
            return result.get("highlights", [])
        else:
            logger.warning(f"Failed to list highlights: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error listing highlights: {e}")
        return []

def display_upload_section():
    """Display video upload section"""
    st.markdown('<h2 class="sub-header">📤 Upload Football Match Video</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a football match video",
            type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'webm'],
            help="Upload a football match video for highlights generation"
        )
        
        if uploaded_file is not None:
            # Display video preview
            st.video(uploaded_file)
            
            # File info
            file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
            st.info(f"**File Info:** {uploaded_file.name} ({file_size:.2f} MB)")
    
    with col2:
        st.markdown("#### 📝 Match Information")
        
        match_info = {
            "team_home": st.text_input("Home Team", "Team A"),
            "team_away": st.text_input("Away Team", "Team B"),
            "competition": st.text_input("Competition", "Premier League"),
            "match_date": st.date_input("Match Date", value=datetime.now().date()),
            "additional_notes": st.text_area("Additional Notes", "")
        }
        
        # Upload and Process button
        if st.button("🚀 Upload & Process", type="primary", use_container_width=True, disabled=uploaded_file is None):
            if uploaded_file:
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Step 1: Upload
                with st.spinner("Uploading video..."):
                    progress_bar.progress(25)
                    upload_result = upload_video(uploaded_file, match_info)
                
                if upload_result and upload_result.get("status") == "success":
                    progress_bar.progress(50)
                    st.success("✅ Video uploaded successfully!")
                    
                    # Get filename from upload result
                    filename = upload_result.get("filename")
                    
                    if filename:
                        # Step 2: Start processing
                        with st.spinner("Starting video processing..."):
                            progress_bar.progress(75)
                            process_result = start_processing(filename=filename)
                        
                        if process_result and "job_id" in process_result:
                            progress_bar.progress(100)
                            job_id = process_result["job_id"]
                            st.success(f"✅ Processing started! Job ID: `{job_id}`")
                            
                            # Store in session state
                            st.session_state.current_job = job_id
                            st.session_state.current_section = "processing"
                            
                            # Add a small delay before rerun
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Failed to start processing!")
                            if process_result:
                                st.error(f"Error: {process_result.get('message', 'Unknown error')}")
                    else:
                        st.error("❌ Could not get filename from upload response!")
                else:
                    st.error("❌ Upload failed!")
                    if upload_result:
                        st.error(f"Error: {upload_result.get('message', 'Unknown error')}")

def display_processing_section():
    """Display video processing section"""
    st.markdown('<h2 class="sub-header">⚙️ Video Processing</h2>', unsafe_allow_html=True)
    
    if "current_job" not in st.session_state or not st.session_state.current_job:
        st.warning("No active processing job. Please upload a video first.")
        if st.button("📤 Go to Upload"):
            st.session_state.current_section = "upload"
            st.rerun()
        return
    
    job_id = st.session_state.current_job
    
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    details_placeholder = st.empty()
    
    # Display initial status
    with status_placeholder.container():
        st.info(f"**Job ID:** `{job_id}`")
        st.markdown("**Status:** ⏳ Checking...")
    
    # Create progress bar
    progress_bar = progress_placeholder.progress(0)
    
    # Poll for status updates
    max_polls = 300  # Max 10 minutes (at 2 second intervals)
    poll_count = 0
    job_completed = False
    
    while not job_completed and poll_count < max_polls:
        poll_count += 1
        
        # Get job status
        status = get_job_status(job_id)
        
        if not status:
            with status_placeholder.container():
                st.error("❌ Failed to get job status. The job may have been cancelled.")
            break
        
        # Update UI
        current_status = status.get("status", "unknown")
        progress = status.get("progress", 0)
        
        # Update progress bar
        progress_bar.progress(progress / 100)
        
        # Update status display
        with status_placeholder.container():
            st.info(f"**Job ID:** `{job_id}`")
            
            # Status with appropriate emoji
            status_emojis = {
                "pending": "⏳",
                "processing": "⚙️",
                "completed": "✅",
                "failed": "❌"
            }
            
            status_emoji = status_emojis.get(current_status, "❓")
            status_text = current_status.upper()
            
            # Color code the status
            if current_status == "completed":
                status_color = "green"
            elif current_status == "failed":
                status_color = "red"
            elif current_status == "processing":
                status_color = "blue"
            else:
                status_color = "orange"
            
            st.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold'>{status_emoji} {status_text}</span>", unsafe_allow_html=True)
            st.markdown(f"**Progress:** {progress}%")
            
            if status.get("processing_time"):
                st.markdown(f"**Processing Time:** {status['processing_time']:.2f} seconds")
        
        # Handle completed job
        if current_status == "completed":
            job_completed = True
            st.balloons()
            
            with details_placeholder.container():
                st.success("🎉 Processing completed successfully!")
                
                # Display results
                if status.get("result"):
                    display_results(status)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📊 View Detailed Results", use_container_width=True):
                        st.session_state.show_detailed_results = True
                        st.rerun()
                with col2:
                    if st.button("🎬 Go to Highlights", use_container_width=True):
                        st.session_state.current_section = "highlights"
                        st.rerun()
                with col3:
                    if st.button("📤 Upload New Video", use_container_width=True):
                        # Clear session state
                        for key in ["current_job", "show_detailed_results"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state.current_section = "upload"
                        st.rerun()
            
            break
        
        # Handle failed job
        elif current_status == "failed":
            job_completed = True
            
            with details_placeholder.container():
                st.error("❌ Processing failed!")
                
                if status.get("error"):
                    st.error(f"**Error:** {status['error']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Retry This Job", use_container_width=True):
                        # We need to get the original filename to retry
                        video_path = status.get("video_path")
                        if video_path:
                            # Extract filename from path
                            filename = Path(video_path).name
                            process_result = start_processing(filename=filename)
                            if process_result and "job_id" in process_result:
                                st.session_state.current_job = process_result["job_id"]
                                st.rerun()
                with col2:
                    if st.button("📤 Upload New Video", use_container_width=True):
                        # Clear session state
                        for key in ["current_job"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state.current_section = "upload"
                        st.rerun()
            
            break
        
        # Wait before next poll
        time.sleep(2)
    
    if not job_completed and poll_count >= max_polls:
        st.error("⏱️ Processing timeout. The job is taking too long.")
        
        if st.button("🔄 Check Again"):
            st.rerun()

def display_results(status):
    """Display processing results"""
    result = status.get("result")
    
    if not result:
        st.warning("No results data available")
        return
    
    st.markdown("### 📊 Processing Results")
    
    # Key metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_events = result.get("total_events", 0)
        st.metric("Events Detected", total_events)
    
    with col2:
        processing_time = status.get("processing_time", 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    with col3:
        output_dir = result.get("output_dir", "Unknown")
        st.text(f"Output: {Path(output_dir).name}")
    
    # Show highlights if available
    highlights_path = result.get("highlights_path")
    if highlights_path:
        st.markdown("#### 🎬 Generated Highlights")
        
        # Check if file exists locally
        if os.path.exists(highlights_path):
            try:
                # Display video
                with open(highlights_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                # Download button
                st.download_button(
                    label="⬇️ Download Highlights",
                    data=video_bytes,
                    file_name=Path(highlights_path).name,
                    mime="video/mp4"
                )
            except Exception as e:
                st.warning(f"Could not load video: {e}")
        else:
            st.info(f"Highlights file: `{highlights_path}`")
    
    # Show events if available
    if "annotations_path" in result:
        annotations_path = result["annotations_path"]
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, 'r') as f:
                    annotations = json.load(f)
                
                events = annotations.get("events", [])
                
                if events:
                    st.markdown(f"#### 🎯 Detected Events ({len(events)} total)")
                    
                    # Event timeline
                    fig = go.Figure()
                    
                    event_types = []
                    event_times = []
                    confidences = []
                    
                    for event in events:
                        event_types.append(event.get("event_type", "Unknown"))
                        event_times.append(event.get("start_time", 0))
                        confidences.append(event.get("confidence", 0.5))
                    
                    fig.add_trace(go.Scatter(
                        x=event_times,
                        y=event_types,
                        mode='markers',
                        marker=dict(
                            size=[c * 20 + 5 for c in confidences],
                            color=confidences,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Confidence"),
                            sizemode='diameter'
                        ),
                        name='Events',
                        hovertemplate="<b>%{y}</b><br>Time: %{x}s<br>Confidence: %{marker.color:.2f}<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        title="Event Timeline",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Event Type",
                        height=400,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Events table
                    st.markdown("#### 📋 Event Details")
                    
                    event_data = []
                    for idx, event in enumerate(events[:20], 1):  # Show first 20 events
                        event_data.append({
                            "#": idx,
                            "Type": event.get("event_type", "Unknown"),
                            "Start": f"{event.get('start_time', 0):.1f}s",
                            "End": f"{event.get('end_time', 0):.1f}s",
                            "Duration": f"{event.get('end_time', 0) - event.get('start_time', 0):.1f}s",
                            "Confidence": f"{event.get('confidence', 0):.1%}",
                            "Description": event.get("description", "")[:50]
                        })
                    
                    if event_data:
                        df = pd.DataFrame(event_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Export button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Export as CSV",
                            data=csv,
                            file_name="detected_events.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.warning(f"Could not load annotations: {e}")

def display_video_library():
    """Display video library section"""
    st.markdown('<h2 class="sub-header">📁 Video Library</h2>', unsafe_allow_html=True)
    
    # Refresh button
    if st.button("🔄 Refresh Library", key="refresh_library"):
        st.rerun()
    
    # List available videos
    videos = list_videos()
    
    if not videos:
        st.info("No videos found. Upload a video to get started.")
        return
    
    st.success(f"Found {len(videos)} video(s)")
    
    # Display videos in a grid
    cols = st.columns(3)
    
    for idx, video in enumerate(videos):
        col_idx = idx % 3
        col = cols[col_idx]
        
        with col:
            # Video card
            video_name = video.get("filename", "Unknown")
            video_size = video.get("size", 0) / (1024 * 1024)  # Convert to MB
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                <div style="font-weight: bold; margin-bottom: 0.5rem;">
                    {video_name[:30]}{'...' if len(video_name) > 30 else ''}
                </div>
                <div style="color: #666; font-size: 0.8rem;">
                    📏 {video_size:.1f} MB
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Process button
            if st.button(f"Process", key=f"process_{video_name}", use_container_width=True):
                with st.spinner(f"Starting processing for {video_name}..."):
                    process_result = start_processing(filename=video_name)
                    
                    if process_result and "job_id" in process_result:
                        job_id = process_result["job_id"]
                        st.success(f"Processing started! Job ID: {job_id}")
                        
                        # Store job ID and switch to processing view
                        st.session_state.current_job = job_id
                        st.session_state.current_section = "processing"
                        st.rerun()
                    else:
                        st.error("Failed to start processing!")

def display_highlights_gallery():
    """Display highlights gallery"""
    st.markdown('<h2 class="sub-header">🎬 Highlights Gallery</h2>', unsafe_allow_html=True)
    
    # Refresh button
    if st.button("🔄 Refresh Highlights", key="refresh_highlights"):
        st.rerun()
    
    # List highlights
    highlights = list_highlights()
    
    if not highlights:
        st.info("No highlights generated yet. Process a video to create highlights.")
        return
    
    st.success(f"Found {len(highlights)} highlight(s)")
    
    # Display highlights
    for highlight in highlights[:10]:  # Show first 10
        with st.expander(f"🎥 {highlight.get('job_id', 'Unknown')}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Try to display video
                highlight_file = highlight.get("highlight_file")
                if highlight_file:
                    try:
                        # Try to construct URL for video
                        job_id = highlight.get("job_id")
                        video_url = f"{API_BASE_URL}/download/{job_id}/{highlight_file}"
                        st.video(video_url)
                    except Exception as e:
                        st.warning(f"Could not load video preview: {e}")
            
            with col2:
                # Highlight info
                st.markdown("**📋 Highlight Info**")
                
                job_id = highlight.get("job_id", "N/A")
                st.text(f"Job ID: {job_id}")
                
                created_time = highlight.get("created", 0)
                if created_time:
                    try:
                        created_str = datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M')
                        st.text(f"Created: {created_str}")
                    except:
                        st.text(f"Created: {created_time}")
                
                # Download button
                if highlight_file and job_id:
                    download_url = f"{API_BASE_URL}/download/{job_id}/{highlight_file}"
                    st.markdown(f'''
                    <a href="{download_url}" target="_blank" style="text-decoration: none;">
                        <button style="
                            background-color: #4CAF50;
                            color: white;
                            padding: 10px 20px;
                            border: none;
                            border-radius: 5px;
                            cursor: pointer;
                            width: 100%;
                            margin-top: 10px;
                        ">
                            ⬇️ Download Highlight
                        </button>
                    </a>
                    ''', unsafe_allow_html=True)

def display_analytics():
    """Display analytics dashboard"""
    st.markdown('<h2 class="sub-header">📈 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Sample analytics data (in real app, fetch from API)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Videos", "15", "+2")
    
    with col2:
        st.metric("Events Detected", "128", "+18")
    
    with col3:
        st.metric("Avg Processing Time", "42s", "-3s")
    
    with col4:
        st.metric("Success Rate", "94%", "+1%")
    
    # Event distribution
    st.markdown("#### 📊 Event Distribution")
    
    event_data = {
        "Goal": 18,
        "Foul": 32,
        "Yellow Card": 12,
        "Red Card": 2,
        "Corner": 28,
        "Free Kick": 24,
        "Penalty": 5,
        "Offside": 7
    }
    
    fig = px.pie(
        names=list(event_data.keys()),
        values=list(event_data.values()),
        hole=0.3,
        title="Types of Events Detected"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application"""
    
    # Initialize session state
    if "current_section" not in st.session_state:
        st.session_state.current_section = "upload"
    
    if "current_job" not in st.session_state:
        st.session_state.current_job = None
    
    if "show_detailed_results" not in st.session_state:
        st.session_state.show_detailed_results = False
    
    # Check API health
    api_healthy = check_api_health()
    
    # Header
    st.markdown('<h1 class="main-header">⚽ Football Highlights Generator</h1>', unsafe_allow_html=True)
    
    # API status indicator
    if api_healthy:
        st.success("✅ API Server: Connected")
    else:
        st.error("❌ API Server: Disconnected")
        st.warning("""
        Please start the API server first:
        
        ```bash
        cd E:\\football-highlights-system
        python -m uvicorn app.main:app --reload --port 8000
        ```
        
        Then refresh this page.
        """)
        
        if st.button("🔄 Check Connection Again"):
            st.rerun()
        
        return  # Stop further execution if API is not available
    
    # Sidebar
    with st.sidebar:
        # Use markdown for the emoji instead of st.image()
        st.markdown("""
        <div style="text-align: center; font-size: 60px; margin-bottom: 10px;">
            ⚽
        </div>
        <div style="text-align: center; font-size: 14px; color: #666; margin-bottom: 20px;">
            Football Highlights Generator
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        # Navigation options
        nav_options = {
            "📤 Upload Video": "upload",
            "⚙️ Processing": "processing",
            "📁 Video Library": "library",
            "🎬 Highlights": "highlights",
            "📈 Analytics": "analytics",
            "⚙️ Settings": "settings"
        }
        
        selected_nav = st.radio(
            "Go to:",
            options=list(nav_options.keys()),
            label_visibility="collapsed",
            key="nav_radio"
        )
        
        # Update current section
        st.session_state.current_section = nav_options[selected_nav]
        
        st.markdown("---")
        
        # Current status
        st.markdown("### 📊 Current Status")
        
        if st.session_state.current_job:
            st.info(f"**Active Job:**\n`{st.session_state.current_job[:15]}...`")
        
        if "uploaded_filename" in st.session_state and st.session_state.uploaded_filename:
            st.info(f"**Current File:**\n`{st.session_state.uploaded_filename[:15]}...`")
        
        st.markdown(f"**API Status:** {'🟢 Connected' if api_healthy else '🔴 Disconnected'}")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True, help="Refresh the page"):
                st.rerun()
        
        with col2:
            if st.button("📊 Stats", use_container_width=True, help="View analytics"):
                st.session_state.current_section = "analytics"
                st.rerun()
        
        if st.button("📤 New Upload", use_container_width=True, type="primary", help="Upload a new video"):
            # Clear previous state
            for key in ["current_job", "uploaded_filename", "show_detailed_results"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_section = "upload"
            st.rerun()
    
    # Main content area
    try:
        current_section = st.session_state.current_section
        
        if current_section == "upload":
            display_upload_section()
            
        elif current_section == "processing":
            display_processing_section()
            
        elif current_section == "library":
            display_video_library()
            
        elif current_section == "highlights":
            display_highlights_gallery()
            
        elif current_section == "analytics":
            display_analytics()
            
        elif current_section == "settings":
            st.markdown('<h2 class="sub-header">⚙️ Settings</h2>', unsafe_allow_html=True)
            
            with st.form("settings_form"):
                st.markdown("#### Processing Settings")
                
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    help="Minimum confidence score for event detection"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    buffer_before = st.number_input(
                        "Buffer Before Event (seconds)",
                        min_value=0,
                        max_value=30,
                        value=5,
                        help="Seconds to include before detected event"
                    )
                
                with col2:
                    buffer_after = st.number_input(
                        "Buffer After Event (seconds)",
                        min_value=0,
                        max_value=30,
                        value=10,
                        help="Seconds to include after detected event"
                    )
                
                st.markdown("#### Model Settings")
                
                model_type = st.selectbox(
                    "Model Type",
                    ["3D CNN", "Multimodal", "Two-Stream"],
                    index=1,
                    help="Type of model to use for event detection"
                )
                
                batch_size = st.selectbox(
                    "Batch Size",
                    [4, 8, 16, 32],
                    index=1,
                    help="Number of samples processed simultaneously"
                )
                
                # Submit button
                submitted = st.form_submit_button("💾 Save Settings", type="primary")
                
                if submitted:
                    # In a real app, save to config or database
                    st.success("✅ Settings saved successfully!")
                    st.info("Note: In production, these settings would be saved to configuration system.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Error in main application")
        
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
    
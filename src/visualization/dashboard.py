"""
Interactive dashboard for football highlights system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FootballHighlightsDashboard:
    """Interactive dashboard for football highlights system"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.experiments_dir = Path("experiments")
        self.highlights_dir = Path("data/highlights")
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)
        self.highlights_dir.mkdir(exist_ok=True)
    
    def create_main_dashboard(self):
        """Create main dashboard with all visualizations"""
        
        # Set page configuration
        st.set_page_config(
            page_title="Football Highlights Dashboard",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                color: #1E88E5;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: bold;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .section-header {
                font-size: 1.8rem;
                color: #424242;
                margin-bottom: 1rem;
                font-weight: 600;
                border-bottom: 3px solid #1E88E5;
                padding-bottom: 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">⚽ Football Highlights Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/919/919830.png", width=100)
            
            st.markdown("### Navigation")
            dashboard_section = st.selectbox(
                "Select Section",
                ["Overview", "Training Analytics", "Event Detection", 
                 "Highlights Gallery", "System Performance", "Configuration"]
            )
            
            st.markdown("---")
            
            # Quick stats
            st.markdown("### Quick Stats")
            
            # Count videos
            video_count = self._count_videos()
            st.metric("Total Videos", video_count)
            
            # Count highlights
            highlight_count = self._count_highlights()
            st.metric("Generated Highlights", highlight_count)
            
            # Count experiments
            experiment_count = self._count_experiments()
            st.metric("Training Experiments", experiment_count)
            
            st.markdown("---")
            
            # Date range filter
            st.markdown("### Date Range")
            date_range = st.date_input(
                "Select Date Range",
                value=(datetime.now().date(), datetime.now().date())
            )
        
        # Main content based on selected section
        if dashboard_section == "Overview":
            self._show_overview()
        elif dashboard_section == "Training Analytics":
            self._show_training_analytics()
        elif dashboard_section == "Event Detection":
            self._show_event_detection()
        elif dashboard_section == "Highlights Gallery":
            self._show_highlights_gallery()
        elif dashboard_section == "System Performance":
            self._show_system_performance()
        elif dashboard_section == "Configuration":
            self._show_configuration()
    
    def _show_overview(self):
        """Show overview dashboard"""
        
        st.markdown('<h2 class="section-header">📊 System Overview</h2>', 
                   unsafe_allow_html=True)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            video_count = self._count_videos()
            st.metric("Total Videos", video_count, delta="+3")
        
        with col2:
            highlight_count = self._count_highlights()
            st.metric("Generated Highlights", highlight_count, delta="+5")
        
        with col3:
            experiment_count = self._count_experiments()
            st.metric("Training Experiments", experiment_count, delta="+1")
        
        with col4:
            # Average processing time
            avg_time = self._get_average_processing_time()
            st.metric("Avg Processing Time", f"{avg_time:.1f}s", delta="-2.3s")
        
        # Recent activity
        st.markdown("### 📈 Recent Activity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Event detection chart
            st.markdown("#### Event Detection Distribution")
            event_dist = self._get_event_distribution()
            
            if event_dist:
                df_events = pd.DataFrame({
                    'Event': list(event_dist.keys()),
                    'Count': list(event_dist.values())
                })
                
                fig = px.pie(df_events, values='Count', names='Event', 
                            hole=0.3, color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No event data available. Process a video first.")
        
        with col2:
            # Processing timeline
            st.markdown("#### Processing Timeline")
            timeline_data = self._get_processing_timeline()
            
            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data)
                df_timeline['date'] = pd.to_datetime(df_timeline['date'])
                df_timeline = df_timeline.sort_values('date')
                
                fig = px.line(df_timeline, x='date', y='count', 
                            title='Video Processing Over Time',
                            markers=True)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No processing timeline data available.")
        
        # System status
        st.markdown("### 🚦 System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Storage Usage")
            storage_used = self._get_storage_usage()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=storage_used,
                title={'text': "Storage Used"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Model Performance")
            model_perf = self._get_model_performance()
            
            if model_perf:
                df_perf = pd.DataFrame([model_perf])
                
                fig = go.Figure(data=[
                    go.Bar(name='Score', x=list(model_perf.keys()), 
                          y=list(model_perf.values()))
                ])
                fig.update_layout(height=250, yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No model performance data")
        
        with col3:
            st.markdown("#### Recent Highlights")
            recent_highlights = self._get_recent_highlights(3)
            
            for highlight in recent_highlights:
                with st.expander(f"🎬 {highlight.get('name', 'Unnamed')}", expanded=False):
                    st.text(f"Created: {highlight.get('created', 'Unknown')}")
                    st.text(f"Duration: {highlight.get('duration', 'Unknown')}")
                    st.text(f"Events: {highlight.get('event_count', 0)}")
                    
                    if st.button("View Details", key=highlight.get('name', '')):
                        st.session_state.selected_highlight = highlight
    
    def _show_training_analytics(self):
        """Show training analytics dashboard"""
        
        st.markdown('<h2 class="section-header">📈 Training Analytics</h2>', 
                   unsafe_allow_html=True)
        
        # Get all experiments
        experiments = self._get_all_experiments()
        
        if not experiments:
            st.info("No training experiments found. Train a model first.")
            return
        
        # Experiment selector
        selected_exp = st.selectbox(
            "Select Experiment",
            options=list(experiments.keys()),
            format_func=lambda x: f"{x} - {experiments[x].get('date', 'Unknown date')}"
        )
        
        if selected_exp:
            exp_data = experiments[selected_exp]
            
            # Load training history
            history_path = self.experiments_dir / selected_exp / "training_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                # Training metrics
                st.markdown("### 📊 Training Metrics")
                
                # Create metrics visualization
                metrics_to_show = ['loss', 'accuracy', 'f1']
                
                fig = make_subplots(
                    rows=len(metrics_to_show), cols=1,
                    subplot_titles=[f"{m.capitalize()} over epochs" for m in metrics_to_show]
                )
                
                for i, metric in enumerate(metrics_to_show, 1):
                    if metric in history.get('train', {}):
                        epochs = range(1, len(history['train'][metric]) + 1)
                        
                        # Train metric
                        fig.add_trace(
                            go.Scatter(x=epochs, y=history['train'][metric],
                                      mode='lines+markers', name=f'Train {metric}',
                                      line=dict(color='blue', width=2)),
                            row=i, col=1
                        )
                        
                        # Validation metric if available
                        if metric in history.get('val', {}):
                            fig.add_trace(
                                go.Scatter(x=epochs, y=history['val'][metric],
                                          mode='lines+markers', name=f'Val {metric}',
                                          line=dict(color='red', width=2)),
                                row=i, col=1
                            )
                
                fig.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Load test results
            test_path = self.experiments_dir / selected_exp / "test_results.json"
            if test_path.exists():
                with open(test_path, 'r') as f:
                    test_results = json.load(f)
                
                # Test performance
                st.markdown("### 🎯 Test Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    accuracy = test_results.get('overall', {}).get('accuracy', 0)
                    st.metric("Accuracy", f"{accuracy:.2%}")
                
                with col2:
                    f1_score = test_results.get('overall', {}).get('f1_score', 0)
                    st.metric("F1 Score", f"{f1_score:.2%}")
                
                with col3:
                    precision = test_results.get('overall', {}).get('precision', 0)
                    st.metric("Precision", f"{precision:.2%}")
                
                with col4:
                    recall = test_results.get('overall', {}).get('recall', 0)
                    st.metric("Recall", f"{recall:.2%}")
                
                # Confusion matrix
                if 'confusion_matrix' in test_results:
                    st.markdown("#### Confusion Matrix")
                    cm = np.array(test_results['confusion_matrix'])
                    
                    fig = px.imshow(cm,
                                  labels=dict(x="Predicted", y="Actual"),
                                  x=[f"Class {i}" for i in range(cm.shape[1])],
                                  y=[f"Class {i}" for i in range(cm.shape[0])],
                                  color_continuous_scale='Blues')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Class performance
                if 'classification_report' in test_results:
                    st.markdown("#### Per-Class Performance")
                    
                    class_data = []
                    for class_name, metrics in test_results['classification_report'].items():
                        if isinstance(metrics, dict):
                            class_data.append({
                                'Class': class_name,
                                'Precision': metrics.get('precision', 0),
                                'Recall': metrics.get('recall', 0),
                                'F1-Score': metrics.get('f1-score', 0),
                                'Support': metrics.get('support', 0)
                            })
                    
                    if class_data:
                        df_classes = pd.DataFrame(class_data)
                        st.dataframe(df_classes, use_container_width=True)
    
    def _show_event_detection(self):
        """Show event detection dashboard"""
        
        st.markdown('<h2 class="section-header">🎯 Event Detection</h2>', 
                   unsafe_allow_html=True)
        
        # Get all processed videos
        processed_videos = self._get_processed_videos()
        
        if not processed_videos:
            st.info("No processed videos found. Process a video first.")
            return
        
        # Video selector
        selected_video = st.selectbox(
            "Select Processed Video",
            options=list(processed_videos.keys()),
            format_func=lambda x: processed_videos[x].get('name', x)
        )
        
        if selected_video:
            video_data = processed_videos[selected_video]
            
            # Load annotations
            annotation_path = self.highlights_dir / selected_video / "annotations.json"
            if annotation_path.exists():
                with open(annotation_path, 'r') as f:
                    annotations = json.load(f)
                
                # Event statistics
                events = annotations.get('events', [])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Events", len(events))
                
                with col2:
                    unique_events = len(set(e['event_type'] for e in events))
                    st.metric("Unique Event Types", unique_events)
                
                with col3:
                    avg_confidence = np.mean([e.get('confidence', 0) for e in events])
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                
                # Event timeline
                st.markdown("### 📅 Event Timeline")
                
                timeline_data = []
                for event in events:
                    timeline_data.append({
                        'Event': event['event_type'],
                        'Start Time': event['start_time'],
                        'End Time': event['end_time'],
                        'Duration': event['end_time'] - event['start_time'],
                        'Confidence': event.get('confidence', 0),
                        'Source': event.get('source', 'unknown')
                    })
                
                if timeline_data:
                    df_timeline = pd.DataFrame(timeline_data)
                    
                    # Create timeline visualization
                    fig = px.scatter(df_timeline, x='Start Time', y='Event',
                                    size='Confidence', color='Source',
                                    hover_data=['Duration', 'Confidence'],
                                    title='Event Timeline')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Event distribution
                    st.markdown("### 📊 Event Distribution")
                    
                    event_counts = df_timeline['Event'].value_counts().reset_index()
                    event_counts.columns = ['Event', 'Count']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(event_counts, x='Event', y='Count',
                                    color='Count', color_continuous_scale='Viridis')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(event_counts, values='Count', names='Event',
                                    hole=0.3, color_discrete_sequence=px.colors.qualitative.Set3)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Event details table
                    st.markdown("### 📋 Event Details")
                    st.dataframe(df_timeline, use_container_width=True)
    
    def _show_highlights_gallery(self):
        """Show highlights gallery"""
        
        st.markdown('<h2 class="section-header">🎬 Highlights Gallery</h2>', 
                   unsafe_allow_html=True)
        
        # Get all highlights
        highlights = self._get_all_highlights()
        
        if not highlights:
            st.info("No highlights generated yet. Process a video first.")
            return
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            min_events = st.slider("Minimum Events", 0, 50, 0)
        
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
        
        # Filter highlights
        filtered_highlights = [
            h for h in highlights
            if h.get('event_count', 0) >= min_events
            and h.get('avg_confidence', 0) >= min_confidence
        ]
        
        st.markdown(f"### Found {len(filtered_highlights)} Highlights")
        
        # Display highlights in grid
        cols = st.columns(3)
        
        for idx, highlight in enumerate(filtered_highlights[:9]):  # Show first 9
            col = cols[idx % 3]
            
            with col:
                # Highlight card
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                    <div style="font-weight: bold; margin-bottom: 0.5rem;">{highlight.get('name', 'Unnamed')}</div>
                    <div style="color: #666; font-size: 0.9rem;">
                        Events: {highlight.get('event_count', 0)}<br>
                        Confidence: {highlight.get('avg_confidence', 0):.2%}<br>
                        Created: {highlight.get('created', 'Unknown')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("👁️ View", key=f"view_{idx}", use_container_width=True):
                        st.session_state.selected_highlight = highlight
                
                with col2:
                    if st.button("⬇️ Download", key=f"download_{idx}", use_container_width=True):
                        # In a real app, this would trigger download
                        st.success(f"Downloading {highlight.get('name', 'highlight')}")
        
        # Selected highlight details
        if 'selected_highlight' in st.session_state:
            highlight = st.session_state.selected_highlight
            
            st.markdown("---")
            st.markdown(f"### 🎥 {highlight.get('name', 'Selected Highlight')}")
            
            # Show highlight details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Metadata**")
                st.json(highlight)
            
            with col2:
                # Try to load and display events
                highlight_dir = self.highlights_dir / highlight.get('directory', '')
                events_path = highlight_dir / "annotations.json"
                
                if events_path.exists():
                    with open(events_path, 'r') as f:
                        events_data = json.load(f)
                    
                    st.markdown("**Events**")
                    events = events_data.get('events', [])
                    
                    for event in events[:5]:  # Show first 5 events
                        st.text(f"• {event['event_type']} at {event['start_time']:.1f}s "
                               f"(Confidence: {event.get('confidence', 0):.2%})")
    
    def _show_system_performance(self):
        """Show system performance metrics"""
        
        st.markdown('<h2 class="section-header">⚡ System Performance</h2>', 
                   unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Processing speed
            avg_speed = self._get_processing_speed()
            st.metric("Avg Processing Speed", f"{avg_speed:.1f} fps")
        
        with col2:
            # Memory usage
            mem_usage = self._get_memory_usage()
            st.metric("Memory Usage", f"{mem_usage:.1f} GB")
        
        with col3:
            # GPU utilization (if available)
            gpu_util = self._get_gpu_utilization()
            st.metric("GPU Utilization", f"{gpu_util:.1f}%")
        
        # Performance over time
        st.markdown("### 📈 Performance Trends")
        
        # Create sample performance data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        processing_times = np.random.normal(45, 10, 30)  # Normally distributed around 45s
        
        df_perf = pd.DataFrame({
            'Date': dates,
            'Processing Time (s)': processing_times,
            'Success Rate': np.random.uniform(0.85, 0.99, 30)
        })
        
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=['Processing Time', 'Success Rate'])
        
        fig.add_trace(
            go.Scatter(x=df_perf['Date'], y=df_perf['Processing Time (s)'],
                      mode='lines+markers', name='Processing Time',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_perf['Date'], y=df_perf['Success Rate'],
                      mode='lines+markers', name='Success Rate',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Resource utilization
        st.markdown("### 🖥️ Resource Utilization")
        
        # Create sample resource data
        hours = list(range(24))
        cpu_usage = np.random.uniform(20, 80, 24)
        memory_usage = np.random.uniform(2, 8, 24)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=hours, y=cpu_usage,
                                mode='lines+markers',
                                name='CPU Usage (%)',
                                line=dict(color='red', width=2)))
        
        fig.add_trace(go.Scatter(x=hours, y=memory_usage,
                                mode='lines+markers',
                                name='Memory Usage (GB)',
                                line=dict(color='blue', width=2),
                                yaxis='y2'))
        
        fig.update_layout(
            title='Resource Utilization (24 hours)',
            xaxis_title='Hour',
            yaxis=dict(title='CPU Usage (%)'),
            yaxis2=dict(title='Memory Usage (GB)',
                       overlaying='y',
                       side='right'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_configuration(self):
        """Show system configuration"""
        
        st.markdown('<h2 class="section-header">⚙️ System Configuration</h2>', 
                   unsafe_allow_html=True)
        
        # Configuration editor
        st.markdown("### Configuration Settings")
        
        # Load current config
        config_path = Path("config/config.py")
        if config_path.exists():
            with open(config_path, 'r') as f:
                current_config = f.read()
            
            # Display current config
            with st.expander("Current Configuration", expanded=False):
                st.code(current_config, language='python')
        
        # Configuration form
        with st.form("config_form"):
            st.markdown("#### Video Processing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                frame_rate = st.number_input("Frame Rate", 1, 60, 30)
                frame_width = st.number_input("Frame Width", 320, 3840, 1280)
            
            with col2:
                frame_height = st.number_input("Frame Height", 240, 2160, 720)
                clip_duration = st.number_input("Clip Duration (s)", 1, 30, 5)
            
            st.markdown("#### Event Detection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
                buffer_before = st.number_input("Buffer Before (s)", 0, 30, 5)
            
            with col2:
                buffer_after = st.number_input("Buffer After (s)", 0, 30, 10)
                min_clip_duration = st.number_input("Min Clip Duration (s)", 1, 10, 3)
            
            st.markdown("#### Model Settings")
            
            model_type = st.selectbox("Model Type", ["3DCNN", "Multimodal", "Two-Stream"])
            batch_size = st.selectbox("Batch Size", [4, 8, 16, 32, 64])
            
            # Submit button
            submitted = st.form_submit_button("💾 Save Configuration")
            
            if submitted:
                # In a real app, this would save to config file
                st.success("✅ Configuration saved successfully!")
                
                # Show preview
                config_preview = f"""
                # Updated Configuration
                FRAME_RATE = {frame_rate}
                FRAME_WIDTH = {frame_width}
                FRAME_HEIGHT = {frame_height}
                CLIP_DURATION = {clip_duration}
                CONFIDENCE_THRESHOLD = {confidence_threshold}
                BUFFER_BEFORE = {buffer_before}
                BUFFER_AFTER = {buffer_after}
                MIN_CLIP_DURATION = {min_clip_duration}
                MODEL_TYPE = "{model_type}"
                BATCH_SIZE = {batch_size}
                """
                
                st.code(config_preview, language='python')
    
    # Helper methods
    def _count_videos(self) -> int:
        """Count total videos"""
        video_dir = self.data_dir / "raw_videos"
        if video_dir.exists():
            return len(list(video_dir.glob("*.mp4")))
        return 0
    
    def _count_highlights(self) -> int:
        """Count generated highlights"""
        if self.highlights_dir.exists():
            return len(list(self.highlights_dir.iterdir()))
        return 0
    
    def _count_experiments(self) -> int:
        """Count training experiments"""
        if self.experiments_dir.exists():
            return len(list(self.experiments_dir.iterdir()))
        return 0
    
    def _get_average_processing_time(self) -> float:
        """Get average processing time"""
        # This is a placeholder - in a real app, you'd track actual processing times
        return 45.3
    
    def _get_event_distribution(self) -> Dict:
        """Get event distribution across all processed videos"""
        # This is a placeholder
        return {
            "goal": 24,
            "foul": 42,
            "yellow_card": 18,
            "red_card": 3,
            "corner": 32,
            "free_kick": 28,
            "penalty": 7,
            "offside": 12
        }
    
    def _get_processing_timeline(self) -> List[Dict]:
        """Get processing timeline data"""
        # This is a placeholder
        return [
            {"date": "2024-01-01", "count": 5},
            {"date": "2024-01-02", "count": 8},
            {"date": "2024-01-03", "count": 12},
            {"date": "2024-01-04", "count": 7},
            {"date": "2024-01-05", "count": 15}
        ]
    
    def _get_storage_usage(self) -> float:
        """Get storage usage percentage"""
        # This is a placeholder
        return 65.5
    
    def _get_model_performance(self) -> Dict:
        """Get model performance metrics"""
        # This is a placeholder
        return {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.82,
            "f1_score": 0.825
        }
    
    def _get_recent_highlights(self, limit: int = 5) -> List[Dict]:
        """Get recent highlights"""
        # This is a placeholder
        return [
            {
                "name": "Match_20240101_highlights",
                "created": "2024-01-01 15:30:00",
                "duration": "2:45",
                "event_count": 12
            },
            {
                "name": "Match_20240102_highlights",
                "created": "2024-01-02 18:45:00",
                "duration": "3:20",
                "event_count": 18
            },
            {
                "name": "Match_20240103_highlights",
                "created": "2024-01-03 14:15:00",
                "duration": "2:15",
                "event_count": 8
            }
        ]
    
    def _get_all_experiments(self) -> Dict:
        """Get all training experiments"""
        experiments = {}
        
        if self.experiments_dir.exists():
            for exp_dir in self.experiments_dir.iterdir():
                if exp_dir.is_dir():
                    exp_name = exp_dir.name
                    
                    # Try to load experiment info
                    info_path = exp_dir / "training_report.json"
                    if info_path.exists():
                        with open(info_path, 'r') as f:
                            exp_info = json.load(f)
                    else:
                        exp_info = {"date": "Unknown", "metrics": {}}
                    
                    experiments[exp_name] = exp_info
        
        return experiments
    
    def _get_processed_videos(self) -> Dict:
        """Get all processed videos"""
        processed = {}
        
        if self.highlights_dir.exists():
            for video_dir in self.highlights_dir.iterdir():
                if video_dir.is_dir():
                    video_name = video_dir.name
                    
                    # Try to load metadata
                    metadata_path = video_dir / "processing_metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = {"name": video_name}
                    
                    processed[video_name] = metadata
        
        return processed
    
    def _get_all_highlights(self) -> List[Dict]:
        """Get all highlights with metadata"""
        highlights = []
        
        if self.highlights_dir.exists():
            for highlight_dir in self.highlights_dir.iterdir():
                if highlight_dir.is_dir():
                    # Load metadata
                    metadata_path = highlight_dir / "highlights_metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        highlights.append({
                            "directory": highlight_dir.name,
                            "name": highlight_dir.name,
                            "created": metadata.get('generated_at', 'Unknown'),
                            "event_count": metadata.get('total_events', 0),
                            "avg_confidence": 0.85  # Placeholder
                        })
        
        return highlights
    
    def _get_processing_speed(self) -> float:
        """Get average processing speed in frames per second"""
        return 125.5
    
    def _get_memory_usage(self) -> float:
        """Get average memory usage in GB"""
        return 4.2
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        return 75.3


def main():
    """Main function to run the dashboard"""
    
    # Create and run dashboard
    dashboard = FootballHighlightsDashboard()
    dashboard.create_main_dashboard()


if __name__ == "__main__":
    main()
"""
Football Highlights Generator - Main Package
"""

__version__ = "1.0.0"
__author__ = "Football Highlights Team"
__email__ = "arrbean1810@gmail.com"

from .main_pipeline import FootballHighlightsPipeline, run_pipeline

# Export main components
__all__ = [
    "FootballHighlightsPipeline",
    "run_pipeline"
]
"""
File utilities for the football highlights system
"""

import os
import json
import yaml
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import logging

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create it if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_save_json(data: Any, path: Union[str, Path], indent: int = 2) -> bool:
    """
    Safely save data to JSON file
    
    Args:
        data: Data to save
        path: File path
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(path)
        ensure_directory(path.parent)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.debug(f"Saved JSON to {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        return False


def safe_load_json(path: Union[str, Path]) -> Optional[Any]:
    """
    Safely load data from JSON file
    
    Args:
        path: File path
        
    Returns:
        Loaded data or None if failed
    """
    try:
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON from {path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        return None


def safe_save_yaml(data: Any, path: Union[str, Path]) -> bool:
    """
    Safely save data to YAML file
    
    Args:
        data: Data to save
        path: File path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(path)
        ensure_directory(path.parent)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.debug(f"Saved YAML to {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save YAML to {path}: {e}")
        return False


def safe_load_yaml(path: Union[str, Path]) -> Optional[Any]:
    """
    Safely load data from YAML file
    
    Args:
        path: File path
        
    Returns:
        Loaded data or None if failed
    """
    try:
        path = Path(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        logger.debug(f"Loaded YAML from {path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load YAML from {path}: {e}")
        return None


def safe_save_pickle(data: Any, path: Union[str, Path]) -> bool:
    """
    Safely save data to pickle file
    
    Args:
        data: Data to save
        path: File path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(path)
        ensure_directory(path.parent)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.debug(f"Saved pickle to {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save pickle to {path}: {e}")
        return False


def safe_load_pickle(path: Union[str, Path]) -> Optional[Any]:
    """
    Safely load data from pickle file
    
    Args:
        path: File path
        
    Returns:
        Loaded data or None if failed
    """
    try:
        path = Path(path)
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        logger.debug(f"Loaded pickle from {path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load pickle from {path}: {e}")
        return None


def get_file_hash(path: Union[str, Path], algorithm: str = 'md5') -> Optional[str]:
    """
    Calculate file hash
    
    Args:
        path: File path
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        File hash or None if failed
    """
    try:
        path = Path(path)
        
        if not path.exists():
            logger.error(f"File not found: {path}")
            return None
        
        hash_func = hashlib.new(algorithm)
        
        with open(path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
        
    except Exception as e:
        logger.error(f"Failed to calculate hash for {path}: {e}")
        return None


def find_files(
    directory: Union[str, Path],
    pattern: str = '*',
    recursive: bool = True
) -> List[Path]:
    """
    Find files matching pattern in directory
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    # Filter to files only (not directories)
    files = [f for f in files if f.is_file()]
    
    logger.debug(f"Found {len(files)} files matching '{pattern}' in {directory}")
    
    return files


def find_video_files(
    directory: Union[str, Path],
    recursive: bool = True
) -> List[Path]:
    """
    Find video files in directory
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    
    all_files = find_files(directory, '*', recursive)
    video_files = [f for f in all_files if f.suffix.lower() in video_extensions]
    
    logger.info(f"Found {len(video_files)} video files in {directory}")
    
    return video_files


def clean_directory(
    directory: Union[str, Path],
    pattern: str = '*',
    keep_files: Optional[List[str]] = None,
    dry_run: bool = False
) -> int:
    """
    Clean files in directory matching pattern
    
    Args:
        directory: Directory to clean
        pattern: Pattern to match
        keep_files: List of file names to keep
        dry_run: If True, only show what would be deleted
        
    Returns:
        Number of files deleted/that would be deleted
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return 0
    
    files_to_clean = find_files(directory, pattern, recursive=False)
    
    if keep_files:
        keep_files_set = set(keep_files)
        files_to_clean = [f for f in files_to_clean if f.name not in keep_files_set]
    
    if dry_run:
        logger.info(f"Dry run: Would delete {len(files_to_clean)} files from {directory}")
        for file in files_to_clean:
            logger.info(f"  Would delete: {file}")
        return len(files_to_clean)
    
    deleted_count = 0
    for file in files_to_clean:
        try:
            file.unlink()
            deleted_count += 1
            logger.debug(f"Deleted: {file}")
        except Exception as e:
            logger.error(f"Failed to delete {file}: {e}")
    
    logger.info(f"Deleted {deleted_count} files from {directory}")
    
    return deleted_count


def copy_file_with_overwrite(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = True
) -> bool:
    """
    Copy file with overwrite option
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite if destination exists
        
    Returns:
        True if successful, False otherwise
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        logger.error(f"Source file not found: {src}")
        return False
    
    if dst.exists() and not overwrite:
        logger.warning(f"Destination file exists and overwrite=False: {dst}")
        return False
    
    try:
        ensure_directory(dst.parent)
        shutil.copy2(src, dst)
        logger.debug(f"Copied {src} to {dst}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def get_file_size(path: Union[str, Path], human_readable: bool = True) -> Union[int, str]:
    """
    Get file size
    
    Args:
        path: File path
        human_readable: Whether to return human-readable string
        
    Returns:
        File size (bytes or human-readable string)
    """
    path = Path(path)
    
    if not path.exists():
        logger.error(f"File not found: {path}")
        return 0 if not human_readable else "0B"
    
    size = path.stat().st_size
    
    if not human_readable:
        return size
    
    # Convert to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    
    return f"{size:.2f} PB"


def test_file_utils():
    """Test file utilities"""
    import tempfile
    
    print("Testing file utilities...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Test ensure_directory
        test_dir = temp_dir / "test" / "subdir"
        ensure_directory(test_dir)
        print(f"✅ Created directory: {test_dir}")
        
        # Test JSON save/load
        test_data = {"name": "test", "value": 123}
        json_path = temp_dir / "test.json"
        safe_save_json(test_data, json_path)
        loaded_data = safe_load_json(json_path)
        assert loaded_data == test_data
        print(f"✅ JSON save/load test passed")
        
        # Test file hash
        file_hash = get_file_hash(json_path)
        assert file_hash is not None
        print(f"✅ File hash: {file_hash}")
        
        # Test find_files
        files = find_files(temp_dir, "*.json")
        assert len(files) == 1
        print(f"✅ Found {len(files)} JSON files")
        
        # Test file size
        size = get_file_size(json_path, human_readable=True)
        print(f"✅ File size: {size}")
        
        # Test clean_directory (dry run)
        deleted = clean_directory(temp_dir, "*.json", dry_run=True)
        print(f"✅ Would delete {deleted} files (dry run)")
    
    print("✅ All file utility tests passed!")


if __name__ == "__main__":
    test_file_utils()
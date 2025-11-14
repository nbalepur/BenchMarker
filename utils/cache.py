import pickle
from pathlib import Path
from typing import Any, Union
from enum import Enum


class CacheType(Enum):
    """Cache behavior types."""
    NONE = "none"           # Never save or load anything
    CACHE = "cache"         # Read if exists, save if doesn't exist
    OVERWRITE = "overwrite" # Always save (overwrite), never read


class Cache:
    """A simple cache object that saves and loads data as pickle files."""
    
    def __init__(self, cache_dir: Union[str, Path], cache_type: CacheType = CacheType.CACHE):
        """Initialize the cache with a directory path and cache type."""
        self.cache_dir = Path(cache_dir)
        self.cache_type = cache_type
        if cache_type != CacheType.NONE:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, id: str, run_name: str, key: str, data: Any) -> None:
        """Save data to cache with the given key."""
        if self.cache_type == CacheType.NONE:
            return  # Never save anything
        
        file_path = self.cache_dir / id / run_name / f"{key}.pkl"
        
        # For CACHE: only save if file doesn't exist
        if self.cache_type == CacheType.CACHE and file_path.exists():
            return  # Don't overwrite existing files
        
        # For OVERWRITE or CACHE (when file doesn't exist): always save
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, id: str, run_name: str, key: str) -> Any:
        """Load data from cache with the given key. Returns None if key doesn't exist."""
        if self.cache_type in {CacheType.NONE, CacheType.OVERWRITE}:
            return None  # Don't load anything
        
        file_path = self.cache_dir / id / run_name / f"{key}.pkl"
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
    
    def clear(self) -> None:
        """Clear all cached files from the directory."""
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file() and file_path.suffix == '.pkl':
                file_path.unlink()

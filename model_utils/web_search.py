import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List
import time
import random
import inspect


class SearchResult:
    """Represents a single search result with URL, content, and metadata."""
    
    def __init__(self, content: str = "", metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}

    def to_citation(self, idx: int) -> str:
        return f"<citation {idx}>\n{self.content}\n</citation {idx}>"


class WebSearch(ABC):
    """Abstract base class for web search implementations."""

    def scrape_page_content(self, url: str, timeout: int = 10) -> str:
        """Default implementation for scraping page content.
        
        Args:
            url: The URL to scrape
            timeout: Request timeout in seconds
            
        Returns:
            Scraped text content from the page
        """
        result = scrape_page_content(url, timeout)
        time.sleep(1 + random.random())
        return result

    def _post_search_sleep(self):
        """Sleep after search operations to avoid rate limiting."""
        time.sleep(1 + random.random())

    @abstractmethod
    def search(self, query: str, max_results: int = 10, **kwargs) -> List[SearchResult]:
        """Search for the given query and return a list of search results.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects containing url, content, and metadata
            
        Note:
            Implementations should call self._post_search_sleep() after completing
            the search operation to avoid rate limiting.
        """
        raise NotImplementedError

# Global flag to control scraping method
SCRAPING_METHOD = "none"  # Options: "tavily", "serper", "beautifulsoup", "none"

def scrape_page_content(url: str, timeout: int = 10) -> str:
    """Extract content from URL using the globally configured scraping method."""
    if SCRAPING_METHOD == "none":
        return _scrape_none(url, timeout)
    elif SCRAPING_METHOD == "tavily":
        return _scrape_with_tavily(url, timeout)
    elif SCRAPING_METHOD == "serper":
        return _scrape_with_serper(url, timeout)
    elif SCRAPING_METHOD == "beautifulsoup":
        return _scrape_with_beautifulsoup(url, timeout)
    else:
        raise ValueError(f"Unknown scraping method: {SCRAPING_METHOD}")

def _scrape_none(url: str, timeout: int = 10) -> str:
    """No scraping - returns empty string."""
    return ""

def _scrape_with_tavily(url: str, timeout: int = 10) -> str:
    """Extract content using Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")
    
    from tavily import TavilyClient
    client = TavilyClient(api_key)
    response = client.extract(
        urls=[url],
        extract_depth="advanced",
    )
    return response.get("results", [{}])[0].get("raw_content", "")

def _scrape_with_serper(url: str, timeout: int = 10) -> str:
    """Extract content using Serper API."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY environment variable is required")
    
    import requests
    serper_url = "https://google.serper.dev/search"
    payload = {
        "url": url
    }
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(serper_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("organic", [{}])[0].get("snippet", "")
    except Exception as e:
        print(f"Error scraping {url} with Serper: {e}")
        return ""

def _scrape_with_beautifulsoup(url: str, timeout: int = 10) -> str:
    """Extract content using BeautifulSoup."""
    from bs4 import BeautifulSoup
    import requests

    try:
        # Add headers to handle compression and encoding properly
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error scraping {url} with BeautifulSoup: {e}")
        return ""

class TavilySearch(WebSearch):
    """Tavily Web Search Implementation"""
    
    def __init__(self, try_scraping: bool = False):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        from tavily import TavilyClient
        self.client = TavilyClient(self.api_key)
        self.try_scraping = try_scraping


    def search(self, query: str, max_results: int = 10, **kwargs) -> Any:
        response = self.client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )
        search_results = []
        for res in response.get("results", []):
            snippet = res.get("content", "")
            scraped_content = ""
            if self.try_scraping and res.get("url", ""):
                scraped_content = self.scrape_page_content(res.get("url", ""))
            
            # Original logic: combine snippet and scraped content for content field
            content = snippet
            if self.try_scraping and scraped_content.strip():
                content += " " + scraped_content
            
            search_results.append(SearchResult(
                content=content.strip(), 
                metadata={
                    'url': res.get("url", ""), 
                    'title': res.get("title", ""),
                    'snippet': snippet,
                    'scraped_page': scraped_content
                }
            ))
        return search_results

class SerperSearch(WebSearch):
    """Serper Web Search Implementation"""
    
    def __init__(self, try_scraping: bool = False):
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("SERPER_API_KEY environment variable is required")
        self.endpoint = "https://google.serper.dev/search"
        self.headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}
        self.try_scraping = try_scraping

    def search(self, query: str, max_results: int = 10, **kwargs) -> Any:
        import requests
        import json
        try:
            payload = json.dumps({"q": query, "autocorrect": False})
            response = requests.request("POST", self.endpoint, headers=self.headers, data=payload)
            response.raise_for_status()
            response_data = response.json()
        except Exception as e:
            print("Error in SerperSearch:", e)
            return []
        search_results = []
        for res in response_data.get("organic", []):
            snippet = res.get("snippet", "")
            scraped_content = ""
            if self.try_scraping and res.get("link", ""):
                scraped_content = self.scrape_page_content(res.get("link", ""))
            
            content = snippet
            if self.try_scraping and scraped_content.strip():
                content += " " + scraped_content
            
            search_results.append(SearchResult(
                content=content.strip(), 
                metadata={
                    'url': res.get("link", ""), 
                    'title': res.get("title", ""),
                    'snippet': snippet,
                    'scraped_page': scraped_content
                }
            ))
        return search_results

class ExaSearch(WebSearch):
    """Exa Web Search Implementation"""

    def __init__(self, try_scraping: bool = False):
        self.api_key = os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError("EXA_API_KEY environment variable is required")
        from exa_py import Exa

        self.client = Exa(api_key = self.api_key)
        self.try_scraping = try_scraping

    def search(self, query: str, max_results: int = 10, **kwargs) -> Any:
        result = self.client.search_and_contents(
            query=query,
            type="auto",
            num_results=max_results,
            include_text=[' '.join(query.replace('"', '').split()[:5])],
            text=True,
            highlights=True,
        )
        search_results = []
        for res in result.results:
            snippet = ' '.join(res.highlights)
            scraped_content = ''
            if self.try_scraping:
                scraped_content = res.text
            
            # Original logic: combine snippet and scraped content for content field
            content = snippet
            if self.try_scraping and scraped_content.strip():
                content += " " + scraped_content
            
            search_results.append(SearchResult(
                content=content.strip(), 
                metadata={
                    'url': res.url, 
                    'title': res.title,
                    'snippet': snippet,
                    'scraped_page': scraped_content
                }
            ))
            print(search_results[-1])
        return search_results

class GoogleSearch(WebSearch):
    """Google Web Search Implementation"""
    
    def __init__(self, try_scraping: bool = False):
        self.api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_CSE_ID")
        self.google_search_url = "https://www.googleapis.com/customsearch/v1"
        self.try_scraping = try_scraping
    
    def search(self, query: str, max_results: int = 10, **kwargs) -> Any:
        import requests

        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": max_results,
        }

        # add extra params
        possible_params = ["lowRange", "highRange"]
        for param in possible_params:
            if kwargs.get(param):
                params[param] = kwargs.get(param)

        time.sleep(1 + random.random())
        response = requests.get(self.google_search_url, params=params)
        try:
            response.raise_for_status()
            response_data = response.json()

            search_results = []
            for item in response_data.get("items", []):
                snippet = item.get('snippet', '')
                scraped_content = ""
                if self.try_scraping:
                    scraped_content = self.scrape_page_content(item['link'])
                
                # Original logic: combine snippet and scraped content for content field
                content = snippet
                if self.try_scraping and scraped_content.strip():
                    content += " " + scraped_content
                
                metadata = {
                    'url': item['link'], 
                    'title': item['title'], 
                    'date': item.get('pagemap', {}).get('metatags', {}),
                    'snippet': snippet,
                    'scraped_page': scraped_content
                }
                search_results.append(SearchResult(content=content.strip(), metadata=metadata))
            return search_results

        except Exception as e:
            print("Error in GoogleSearch:", e)
            return []

class BraveSearch(WebSearch):
    """Brave Search API Implementation - uses requests (no extra package needed)"""
    
    def __init__(self, try_scraping: bool = False):
        self.api_key = os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY environment variable is required")
        self.brave_search_url = "https://api.search.brave.com/res/v1/web/search"
        self.try_scraping = try_scraping
    
    def search(self, query: str, max_results: int = 10, **kwargs) -> Any:
        import requests
        try:
            response = requests.get(
                self.brave_search_url,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "x-subscription-token": self.api_key
                },
                params={"q": query}
            ).json()

            search_results = []
            for result in response.get("web", {}).get("results", []):
                snippet = result.get("description", "")
                scraped_content = ""
                if self.try_scraping:
                    scraped_content = self.scrape_page_content(result["url"])
                
                # Original logic: combine snippet and scraped content for content field
                content = snippet
                if self.try_scraping and scraped_content.strip():
                    content += " " + scraped_content

                search_results.append(SearchResult(
                    content=content.strip(), 
                    metadata={
                        'url': result["url"], 
                        'title': result["title"], 
                        'date': result.get("page_age", ""), 
                        'last_updated': result.get("page_age", ""),
                        'snippet': snippet,
                        'scraped_page': scraped_content
                    }
                ))
            return search_results
        except Exception as e:
            print("Error in BraveSearch:", e)
            return []

class PerplexitySearch(WebSearch):
    """Perplexity Search Implementation - requires perplexity package"""
    
    def __init__(self, try_scraping: bool = False):
        try:
            from perplexity import Perplexity
            self.client = Perplexity()
        except ImportError:
            raise ImportError(
                "PerplexitySearch requires the 'perplexityai' package. "
                "Install it with: pip install mcqa-bench[perplexity]"
            )
        self.try_scraping = try_scraping
    
    def search(self, query: str, max_results: int = 10, max_tokens_per_page: int = 512, **kwargs) -> Any:
        search = self.client.search.create(
            query=query,
            max_results=max_results,
            max_tokens_per_page=max_tokens_per_page
        )
        search_results = []
        for result in search.results:
            snippet = result.snippet
            scraped_content = ""
            if self.try_scraping:
                scraped_content = self.scrape_page_content(result.url)
            
            # Original logic: combine snippet and scraped content for content field
            content = snippet
            if self.try_scraping and scraped_content.strip():
                content += " " + scraped_content
            
            search_results.append(SearchResult(
                content=content, 
                metadata={
                    'url': result.url, 
                    'title': result.title, 
                    'date': result.date, 
                    'last_updated': result.last_updated,
                    'snippet': snippet,
                    'scraped_page': scraped_content
                }
            ))
        return search_results


class WebSearchType(Enum):
    """Enum for different types of web search implementations."""
    GOOGLE = "google"
    BRAVE = "brave"
    PERPLEXITY = "perplexity"
    EXA = "exa"
    TAVILY = "tavily"
    SERPER = "serper"

def _get_valid_params(cls) -> set:
    """Get valid parameter names for a class constructor, excluding 'self'."""
    sig = inspect.signature(cls.__init__)
    return set(sig.parameters.keys()) - {'self'}


def _create_with_filtered_kwargs(cls, search_type_name: str, **kwargs) -> WebSearch:
    """Create a WebSearch instance with filtered kwargs and warnings for unused parameters."""
    valid_params = _get_valid_params(cls)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    unused = set(kwargs.keys()) - valid_params
    
    if unused:
        print(f"Warning: Unused parameters for {search_type_name}: {', '.join(unused)}")
    
    return cls(**filtered_kwargs)


def create_web_searcher(search_type: WebSearchType, **kwargs) -> WebSearch:
    """Factory function to create a WebSearch implementation based on the enum.
    
    Args:
        search_type: The type of web searcher to create
        **kwargs: Additional arguments to pass to the searcher constructor
        
    Returns:
        A WebSearch implementation instance
        
    Raises:
        ValueError: If the search_type is not supported
        ImportError: If required dependencies are not installed
    """
    if search_type == WebSearchType.GOOGLE:
        return _create_with_filtered_kwargs(GoogleSearch, "GoogleSearch", **kwargs)
    if search_type == WebSearchType.BRAVE:
        return _create_with_filtered_kwargs(BraveSearch, "BraveSearch", **kwargs)
    if search_type == WebSearchType.PERPLEXITY:
        return _create_with_filtered_kwargs(PerplexitySearch, "PerplexitySearch", **kwargs)
    if search_type == WebSearchType.EXA:
        return _create_with_filtered_kwargs(ExaSearch, "ExaSearch", **kwargs)
    if search_type == WebSearchType.TAVILY:
        return _create_with_filtered_kwargs(TavilySearch, "TavilySearch", **kwargs)
    if search_type == WebSearchType.SERPER:
        return _create_with_filtered_kwargs(SerperSearch, "SerperSearch", **kwargs)
    raise ValueError(f"Unsupported WebSearchType: {search_type}")



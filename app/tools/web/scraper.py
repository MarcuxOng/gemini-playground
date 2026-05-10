"""
Web Scraper tool — fetches and extracts text from URLs.
"""

from __future__ import annotations

import logging
import requests
import socket
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from app.tools import register

logger = logging.getLogger(__name__)

def is_safe_url(url: str) -> bool:
    """
    Check if a URL is safe to fetch (prevents SSRF).
    Blocks private IP ranges, localhost, and non-http/https protocols.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Resolve hostname with socket.getaddrinfo to cover all records (IPv4/IPv6)
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except (OSError, socket.gaierror):
            # Handle specific DNS resolution errors
            return False

        for family, _, _, _, sockaddr in addr_info:
            ip_address = str(sockaddr[0])
            
            if family == socket.AF_INET:
                # IPv4 validation
                try:
                    ip_parts = list(map(int, ip_address.split('.')))
                    if len(ip_parts) != 4:
                        return False
                    
                    # 127.0.0.0/8 (Loopback)
                    if ip_parts[0] == 127:
                        return False
                    # 10.0.0.0/8 (Private)
                    if ip_parts[0] == 10:
                        return False
                    # 172.16.0.0/12 (Private)
                    if ip_parts[0] == 172 and (16 <= ip_parts[1] <= 31):
                        return False
                    # 192.168.0.0/16 (Private)
                    if ip_parts[0] == 192 and ip_parts[1] == 168:
                        return False
                    # 169.254.0.0/16 (Link-local)
                    if ip_parts[0] == 169 and ip_parts[1] == 254:
                        return False
                except (ValueError, IndexError):
                    return False
            
            elif family == socket.AF_INET6:
                # IPv6 validation
                # Block ::1 (loopback)
                if ip_address == '::1':
                    return False
                # Block fe80::/10 (link-local)
                if ip_address.lower().startswith('fe80:'):
                    return False
                # Block unique local addresses (fc00::/7)
                if ip_address.lower().startswith('fc') or ip_address.lower().startswith('fd'):
                    return False
        
        return True
    except Exception:
        return False


@register
def scrape_url(url: str, max_chars: int = 4000) -> str:
    """
    Fetch a URL and extract clean text from it. 
    Use this when the user provides a direct link or you need more context than a search summary.

    :param url: The full URL to scrape (e.g., 'https://example.com').
    :param max_chars: The maximum amount of text to return (default 4000).
    """
    try:
        current_url = url
        max_redirects = 5
        redirect_count = 0

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }

        while redirect_count <= max_redirects:
            if not is_safe_url(current_url):
                return f"Error: The URL '{current_url}' is restricted for security reasons (SSRF protection)."

            logger.info(f"Fetching URL: {current_url}")
            response = requests.get(current_url, headers=headers, timeout=15, allow_redirects=False)
            
            # Handle redirects manually to ensure safety at each hop
            if 300 <= response.status_code < 400:
                redirect_location = response.headers.get('Location')
                if not redirect_location:
                    break
                
                from urllib.parse import urljoin
                current_url = urljoin(current_url, redirect_location)
                redirect_count += 1
            else:
                break
        
        if redirect_count > max_redirects:
            return f"Error: Too many redirects ({redirect_count}) for URL: {url}"

        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script, style, and other noise
        for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
            tag.decompose()
            
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        
        if not clean_text:
            return f"The URL {current_url} was fetched, but no readable text was found."

        # Limit response length
        if len(clean_text) > max_chars:
            return clean_text[:max_chars] + f"\n\n[...Truncated to {max_chars} chars...]"
            
        return clean_text

    except Exception as e:
        logger.error(f"Scraper error for {url}: {e}")
        return f"Error scraping the URL: {str(e)}"

# SRag (Smart RAG)

SRag is a Python-based **smart web search + scraping layer** designed to act like a local, self-hosted alternative to Tavily-style search APIs, optimized for LLM / RAG workflows.

It uses DuckDuckGo (DDGS) for discovery, async HTTP requests for speed, and HTML cleaning for LLM-friendly text output.

---

## Features

- 🔎 **Web search for LLMs**
  - Uses DDGS to find top results for a query.
  - Returns structured results with `title`, `url`, `content`, `published_date`, `author`, and `score`.

- ⚡ **Async scraping**
  - Uses `httpx.AsyncClient` for parallel fetching.
  - Keeps hop latency low by scraping multiple URLs concurrently.

- 🧹 **Content cleaning**
  - Parses HTML with BeautifulSoup.
  - Removes noisy elements (`nav`, `footer`, `script`, `style`).
  - Produces clean plain text suitable for LLM context.

- 🧠 **Tavily-like API**
  - `search(query)` method returns a list of dicts ready to plug into an LLM tool.
  - Each result includes both a trimmed `content` and full `raw_content`.

- 🏷️ **Metadata extraction**
  - Extracts title, published date (when available), author, and Open Graph image.

---

## Project Status

This is an early prototype of SRag (Smart RAG):

- ✅ Working async search + scrape + clean pipeline  
- ✅ Simple Tavily-style `search()` API  
- ✅ CLI demo for quick testing  

Planned next steps:

- Optional advanced content extraction (e.g., `trafilatura` / Crawl4AI)  
- Vector store integration (LanceDB, sqlite-vec, etc.)  
- FastAPI HTTP service (`/search`) for easy integration with agents  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/square-box-hash/srag.git
cd srag

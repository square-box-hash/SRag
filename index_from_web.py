import sys
import os
import asyncio

# --- PATH HELPER: Fixes 'ModuleNotFoundError' ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
try:
    from srag.scraper import AnuInfrastructureScraper
except ImportError:
    from srag import AnuInfrastructureScraper

from srag.indexer import SRagIndexer

async def run_deep_research(query: str, session_id: str):
    """
    Performs multi-URL research and saves it to a unique table.
    """
    # 1. Initialize the Scraper with high capacity (10+ results)
    scraper = AnuInfrastructureScraper(
        max_results=12,      # Increased for deeper research
        max_chars=2000,       # More context per page
        extract_mode="trafilatura",
    )
    
    print(f"\n🚀 Starting Deep Research")
    print(f"🔍 Query: {query}")
    print(f"📁 Session/Table: {session_id}")
    print(f"--- Searching and Scraping multiple sources... ---")
    
    # 2. Get the facts (async)
    # This hits the web and processes results in parallel
    docs = await scraper.get_facts(query)

    if not docs:
        print("❌ No documents were found. Check your connection or scraper settings.")
        return

    # 3. Initialize the Indexer
    indexer = SRagIndexer()
    
    # 4. Index into a specific table
    # force_new=True automatically clears the old schema/data for THIS session only
    indexer.index_documents(
        docs, 
        table_name=session_id, 
        force_new=True 
    )
    
    print(f"✅ Success! Indexed {len(docs)} documents into table '{session_id}'.")
    print(f"💡 You can now search this specific context in your demo/API.")

if __name__ == "__main__":
    # --- LLM-FRIENDLY CONFIGURATION ---
    # In a real app, these would come from your API request or LLM logic
    USER_QUERY = "current marriage laws 2026"
    SESSION = "marriage_laws_research" 

    asyncio.run(run_deep_research(USER_QUERY, SESSION))
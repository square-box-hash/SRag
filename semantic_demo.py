import sys
import os

# --- PATH HELPER: Fixes 'ModuleNotFoundError' ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from srag.indexer import SRagIndexer

def main():
    # 1. Connect to the existing DB
    # (Make sure you run index_from_web.py first!)
    try:
        indexer = SRagIndexer()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return

    query = "how to query nested jsonb fields"
    print(f"\n--- Running Semantic Search for: '{query}' ---")

    try:
        # 2. Perform the search
        results = indexer.semantic_search(query, k=3)

        if not results:
            print("No matches found in the database.")
            return

        # 3. Display the results
        for i, r in enumerate(results, start=1):
            title = r.get("title", "No Title")
            url = r.get("url", "No URL")
            content = r.get("content", "")
            
            # _distance is the similarity score from LanceDB
            score = r.get("_distance", 0)
            
            print(f"[{i}] {title}")
            print(f"    Source: {url}")
            print(f"    Relevance: {max(0, 100 - (score * 100)):.1f}%")
            print(f"    Text: {content[:250]}...")
            print("-" * 40)

    except Exception as e:
        print(f"Search Error: {e}")
        print("Tip: If you get a 'No vector column' error, delete the 'srag_db' folder and re-run index_from_web.py")

if __name__ == "__main__":
    main()

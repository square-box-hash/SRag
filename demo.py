import asyncio
from srag import AnuInfrastructureScraper

async def main():
    scraper = AnuInfrastructureScraper(max_results=3, max_chars=1500)
    results = await scraper.search("postgresql jsonb tutorial")

    for i, r in enumerate(results, start=1):
        print(f"[{i}] {r['title']}")
        print(r["url"])
        print(r["content"][:300], "\n---\n")

if __name__ == "__main__":
    asyncio.run(main())

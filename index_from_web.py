import asyncio

from srag import AnuInfrastructureScraper
from srag.indexer import SRagIndexer


async def main():
    scraper = AnuInfrastructureScraper(
        max_results=3,
        max_chars=1500,
        extract_mode="trafilatura",
    )
    docs = await scraper.get_facts("postgresql jsonb tutorial")

    indexer = SRagIndexer()
    indexer.index_documents(docs)
    print(f"Indexed {len(docs)} documents into LanceDB.")


if __name__ == "__main__":
    asyncio.run(main())

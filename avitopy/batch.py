"""
Batch-раннер: читает запросы из input.json и прогоняет скрапер по каждому.
Аналог run_all.js.

Запуск:
  python batch.py
  python batch.py --input queries.json --output ./results --no-headless
"""

import argparse
import json
import os
from urllib.parse import quote
from scraper import Scraper


def build_avito_url(query: str) -> str:
    return f"https://www.avito.ru/all/vakansii?cd=1&q={quote(query)}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch Avito scraper — обходит все запросы из input.json"
    )
    parser.add_argument(
        "--input",
        default="input.json",
        help="Путь к JSON с запросами (по умолчанию: input.json)",
    )
    parser.add_argument(
        "--output",
        default="./results",
        help="Директория для результатов (по умолчанию: ./results)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Показывать браузер",
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data.get("queries", [])
    if not queries:
        print("No queries found in input file.")
        return

    os.makedirs(args.output, exist_ok=True)

    scraper = Scraper()
    for i, entry in enumerate(queries):
        q = entry["query"]
        file_name = f"query_{i}.json"
        print(f"\n=== [{i + 1}/{len(queries)}] Query: \"{q}\" ===")

        scraper.scrape(
            {
                "url": build_avito_url(q),
                "outputPath": args.output,
                "fileName": file_name,
                "pages": 0,
                "headless": not args.no_headless,
            }
        )

    print("\nAll queries done.")


if __name__ == "__main__":
    main()

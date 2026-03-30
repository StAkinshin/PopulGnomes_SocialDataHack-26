"""
CLI-обёртка для скрапера Avito.
Аналог index.js (yargs) — используем argparse.

Примеры запуска:
  python cli.py -u "https://www.avito.ru/all/vakansii?q=рабочий+дом"
  python cli.py -u "https://www.avito.ru/all/vakansii?q=рабочий+дом" -p 3 -d ./results -f out.json
  python cli.py -u "..." --no-headless   # чтобы видеть браузер
"""

import argparse
import os
from scraper import Scraper


def main():
    parser = argparse.ArgumentParser(
        description="Avito scraper (undetected-chromedriver)"
    )
    parser.add_argument(
        "-u", "--url",
        required=True,
        help="URL страницы Avito для скрапинга",
    )
    parser.add_argument(
        "-d", "--directory",
        default=os.getcwd(),
        help="Путь к директории для результатов (по умолчанию: текущая)",
    )
    parser.add_argument(
        "-f", "--file",
        default="adverts.json",
        help="Имя выходного JSON-файла (по умолчанию: adverts.json)",
    )
    parser.add_argument(
        "-p", "--pages",
        type=int,
        default=0,
        help="Количество страниц (0 = все)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Показывать браузер (по умолчанию headless)",
    )

    args = parser.parse_args()

    scraper = Scraper()
    scraper.scrape(
        {
            "url": args.url,
            "outputPath": args.directory,
            "fileName": args.file,
            "pages": args.pages,
            "headless": not args.no_headless,
        }
    )


if __name__ == "__main__":
    main()

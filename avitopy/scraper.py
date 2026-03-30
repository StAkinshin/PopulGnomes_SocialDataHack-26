"""
Avito scraper — Python + undetected-chromedriver (Selenium).
Аналог scraper.js (Puppeteer) с параллельным сбором описаний.
"""

import json
import time
import random
import os
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



def random_delay(min_sec: float, max_sec: float):
    time.sleep(min_sec + random.random() * (max_sec - min_sec))


def make_chrome(headless: bool = True) -> uc.Chrome:
    """Создаёт экземпляр undetected-chromedriver."""
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    driver = uc.Chrome(options=options, use_subprocess=True,version_main=146)
    driver.set_page_load_timeout(120)
    return driver


class Scraper:
    def __init__(self):
        self.driver: uc.Chrome | None = None
        self.output_file = None
        self.first_advert = True
        self.headless = True
        self._current_url = ""  # запоминаем URL для перезахода после рестарта

    # ------------------------------------------------------------------ #
    #  Обработка капчи и блокировок                                        #
    # ------------------------------------------------------------------ #
    MAX_CAPTCHA_RETRIES = 3

    def _page_has_content(self) -> bool:
        """Проверяет, есть ли на странице реальный контент Avito."""
        if not self.driver:
            return False
        for selector in [
            'div[data-marker="catalog-serp"]',
            'div[itemprop="description"]',
            'div[class*="description-root"]',
            'div[data-marker="item-view"]',
        ]:
            try:
                self.driver.find_element(By.CSS_SELECTOR, selector)
                return True
            except Exception:
                pass
        return False

    def _restart_browser(self):
        """Полностью убивает и поднимает браузер заново (новый fingerprint)."""
        print("  [block] Restarting browser for new fingerprint...")
        self.close_browser()
        random_delay(5.0, 10.0)
        self.start_browser(headless=self.headless)

    def handle_captcha(self, url: str = "") -> bool:
        """
        Проверяет наличие капчи или блок-страницы.
        Возвращает True если удалось пройти, False если нет.
        """
        if not self.driver:
            return False

        for attempt in range(self.MAX_CAPTCHA_RETRIES):
            # Тип 1: кнопка "Продолжить"
            try:
                btn = self.driver.find_element(
                    By.CSS_SELECTOR, 'div.form-action button[name="submit"]'
                )
                print(f"  [captcha] Found 'Продолжить' button, clicking... (attempt {attempt + 1})")
                random_delay(1.0, 3.0)
                btn.click()
                random_delay(3.0, 6.0)

                if self._page_has_content():
                    return True
                # Не помогло — продолжаем цикл
                continue
            except Exception:
                pass

            # Контент есть — всё ок
            if self._page_has_content():
                return True

            # Тип 2: блок-стена — рестарт браузера + перезаход на URL
            wait_sec = 15 + attempt * 20  # 15, 35, 55 секунд
            print(
                f"  [block] Wall detected! Waiting {wait_sec}s, "
                f"then restarting browser... (attempt {attempt + 1}/{self.MAX_CAPTCHA_RETRIES})"
            )
            time.sleep(wait_sec)

            self._restart_browser()

            if url:
                try:
                    self.driver.get(url)
                    random_delay(2.0, 4.0)

                    if self._page_has_content():
                        print("  [block] Bypass successful after restart!")
                        return True
                except Exception as err:
                    print(f"  [block] Error after restart: {err}")

        print("  [block] Could not bypass block after all retries")
        return False

    # ------------------------------------------------------------------ #
    #  Браузер                                                            #
    # ------------------------------------------------------------------ #
    def start_browser(self, headless: bool = True):
        self.headless = headless
        self.driver = make_chrome(headless=headless)

    def close_browser(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    # ------------------------------------------------------------------ #
    #  Сбор полного описания с отдельной страницы объявления               #
    # ------------------------------------------------------------------ #
    def fetch_full_description(self, url: str) -> str:
        if not self.driver:
            return ""
    
        original_window = self.driver.current_window_handle  # запомнить
    
        try:
            self.driver.execute_script("window.open('');")
            new_window = [w for w in self.driver.window_handles 
                          if w != original_window][-1]
            self.driver.switch_to.window(new_window)
            self.driver.get(url)
        
            random_delay(0.5, 1.0)
            self.handle_captcha(url=url)
        
            description = ""
            try:
                el = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, 'div[itemprop="description"]')
                    )
                )
                description = (el.text or "").strip()
            except Exception:
                try:
                    el = self.driver.find_element(
                        By.CSS_SELECTOR, 'div[class*="description-root"]'
                    )
                    description = (el.text or "").strip()
                except Exception:
                    pass
        
            return description
    
        except Exception as err:
            print(f"Could not fetch description from {url}: {err}")
            return ""
    
        finally:
            try:
            # закрыть только если это НЕ единственная вкладка
                if len(self.driver.window_handles) > 1:
                    self.driver.close()
                self.driver.switch_to.window(original_window)
            except Exception:
                pass

    def enrich_adverts(self, adverts: list[dict]):
        """Последовательно загружает описания для списка объявлений."""
        for advert in adverts:
            if advert.get("url"):
                advert["description"] = self.fetch_full_description(advert["url"])

    # ------------------------------------------------------------------ #
    #  Извлечение объявлений со страницы листинга                          #
    # ------------------------------------------------------------------ #
    def extract_adverts_from_page(self, query: str) -> list[dict]:
        """
        Парсит карточки объявлений на текущей странице.
        Использует Selenium вместо Puppeteer $$eval.
        """
        if not self.driver:
            return []

        items = self.driver.find_elements(
            By.CSS_SELECTOR, 'div[data-marker="item"]'
        )
        adverts = []
        for item in items:
            try:
                advert_id = item.get_attribute("data-item-id") or ""

                # Заголовок и ссылка
                title = ""
                url = ""
                try:
                    title_el = item.find_element(
                        By.CSS_SELECTOR, 'a[data-marker="item-title"]'
                    )
                    title = (title_el.text or "").strip()
                    href = title_el.get_attribute("href") or ""
                    if href:
                        if href.startswith("http"):
                            url = href.split("?")[0]
                        else:
                            url = "https://www.avito.ru" + href.split("?")[0]
                except Exception:
                    pass

                # Цена
                price = 0
                try:
                    price_el = item.find_element(
                        By.CSS_SELECTOR, 'span[data-marker="item-price-value"]'
                    )
                    price_text = (price_el.text or "").replace(" ", "").replace("\xa0", "")
                    digits = "".join(c for c in price_text if c.isdigit())
                    price = int(digits) if digits else 0
                except Exception:
                    pass

                # Мета-описание (короткое)
                description = ""
                try:
                    desc_meta = item.find_element(
                        By.CSS_SELECTOR, 'meta[itemprop="description"]'
                    )
                    description = (desc_meta.get_attribute("content") or "").strip()
                except Exception:
                    pass

                # Продавец
                author = ""
                seller_id = ""
                try:
                    seller_link = item.find_element(
                        By.CSS_SELECTOR, 'div[class*="iva-item-sellerInfo"] a'
                    )
                    try:
                        name_p = seller_link.find_element(By.TAG_NAME, "p")
                        author = (name_p.text or "").strip()
                    except Exception:
                        pass
                    seller_href = seller_link.get_attribute("href") or ""
                    if "/brands/" in seller_href:
                        seller_id = seller_href.split("/brands/")[1].split("?")[0]
                except Exception:
                    pass

                # Локация
                address = ""
                metro = ""
                try:
                    loc_el = item.find_element(
                        By.CSS_SELECTOR, 'div[data-marker="item-location"]'
                    )
                    try:
                        pin = loc_el.find_element(
                            By.CSS_SELECTOR, 'span[class*="geo-pinIcon"]'
                        )
                        parent = pin.find_element(By.XPATH, "..")
                        address = (parent.text or "").strip()
                    except Exception:
                        pass
                    try:
                        metro_icon = loc_el.find_element(
                            By.CSS_SELECTOR, 'span[class*="geo-icons"]'
                        )
                        station = metro_icon.find_element(
                            By.XPATH, "following-sibling::*[1]"
                        )
                        metro = (station.text or "").strip()
                    except Exception:
                        pass
                except Exception:
                    pass

                adverts.append(
                    {
                        "advertId": advert_id,
                        "query": query,
                        "title": title,
                        "description": description,
                        "url": url,
                        "price": price,
                        "author": author,
                        "sellerId": seller_id,
                        "address": address,
                        "metro": metro,
                        "date": "",
                        "phone": "",
                    }
                )
            except Exception as err:
                print(f"Error parsing item: {err}")
                continue

        return adverts

    # ------------------------------------------------------------------ #
    #  Пагинация                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_page_url(base_url: str, page: int) -> str:
        parsed = urlparse(base_url)
        params = parse_qs(parsed.query, keep_blank_values=True)
        if page > 1:
            params["p"] = [str(page)]
        else:
            params.pop("p", None)
        new_query = urlencode(params, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    def iter_pages(self, url: str, query: str):
        """Генератор: отдаёт список объявлений для каждой страницы."""
        if not self.driver:
            raise RuntimeError("Browser not started")

        page_num = 1
        while True:
            page_url = self.build_page_url(url, page_num)
            print(f"Scraping page {page_num}: {page_url}")

            self.driver.get(page_url)

            # Проверяем капчу/блокировку
            random_delay(1.0, 2.0)
            if not self.handle_captcha(url=page_url):
                print(f"Blocked on page {page_num}, skipping.")
                break

            # Ждём появления каталога
            try:
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, 'div[data-marker="catalog-serp"]')
                    )
                )
            except Exception:
                page_title = self.driver.title
                debug_path = os.path.join(
                    os.getcwd(), f"debug_page{page_num}.png"
                )
                try:
                    self.driver.save_screenshot(debug_path)
                except Exception:
                    pass
                print(
                    f'No items found on page {page_num} '
                    f'(title: "{page_title}"). Screenshot: {debug_path}'
                )
                break

            adverts = self.extract_adverts_from_page(query)
            if not adverts:
                page_title = self.driver.title
                debug_path = os.path.join(
                    os.getcwd(), f"debug_page{page_num}_empty.png"
                )
                try:
                    self.driver.save_screenshot(debug_path)
                except Exception:
                    pass
                print(
                    f'No adverts on page {page_num} '
                    f'(title: "{page_title}"). Screenshot: {debug_path}'
                )
                break

            yield adverts
            page_num += 1
            random_delay(2.0, 5.0)

    # ------------------------------------------------------------------ #
    #  Запись в JSON                                                      #
    # ------------------------------------------------------------------ #
    def write_advert(self, advert: dict):
        if not self.output_file:
            return
        prefix = "\n" if self.first_advert else ",\n"
        self.output_file.write(prefix + json.dumps(advert, ensure_ascii=False, indent=2))
        self.first_advert = False

    # ------------------------------------------------------------------ #
    #  Основной цикл скрапинга                                            #
    # ------------------------------------------------------------------ #
    def scrape_adverts(
        self, url: str, query: str, output_path: str, file_name: str, pages: int
    ):
        count = 0
        try:
            os.makedirs(output_path, exist_ok=True)
            filepath = os.path.join(output_path, file_name)
            self.output_file = open(filepath, "w", encoding="utf-8")
            self.output_file.write("[")

            for adverts in self.iter_pages(url, query):
                # Последовательный сбор описаний через вкладки
                self.enrich_adverts(adverts)

                for advert in adverts:
                    self.write_advert(advert)

                print(f"Page {count + 1}: got {len(adverts)} adverts")
                count += 1
                if pages > 0 and count >= pages:
                    break

        except Exception as err:
            print(f"Got an error while scraping: {err}")

        if self.output_file:
            self.output_file.write("\n]\n")
            self.output_file.close()
            self.output_file = None

    # ------------------------------------------------------------------ #
    #  Точка входа                                                        #
    # ------------------------------------------------------------------ #
    def scrape(self, options: dict):
        url = options.get("url", "")
        output_path = options.get("outputPath", os.getcwd())
        file_name = options.get("fileName", "adverts.json")
        pages = options.get("pages", 0)
        headless = options.get("headless", True)

        if not url:
            raise ValueError("No url to scrape.")

        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        query = params.get("q", [url])[0]

        self.first_advert = True

        try:
            self.start_browser(headless=headless)
            self.scrape_adverts(url, query, output_path, file_name, pages)
        except Exception as err:
            print(f"Got an error while scraping: {err}")
        finally:
            self.close_browser()
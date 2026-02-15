import os
import time
import re
import json
import vk_api
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# настройки
load_dotenv()
VK_TOKEN = os.getenv("VK_TOKEN")
INPUT_FILENAME = "input_vk.json"

# Лимиты
# ВК отдает максимум 1000 записей на один запрос.
MAX_POSTS_PER_QUERY = 200    # Сколько постов собирать на каждый запрос из JSON 
DAYS_TO_CHECK = 365          # Не старше года

def extract_phone(text):
    if not text: return None
    pattern = r'(?:\+7|8|7)[\s\-]?\(?(\d{3})\)?[\s\-]?(\d{3})[\s\-]?(\d{2})[\s\-]?(\d{2})'
    match = re.search(pattern, text)
    if match: return match.group(0)
    return None

def main():
    print("Запуск поиска ...")
    
    if not VK_TOKEN:
        print("Ошибка: Токен не найден в .env")
        return

    if not os.path.exists(INPUT_FILENAME):
        print(f"Файл {INPUT_FILENAME} не найден")
        return

    with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = [item['query'] for item in data['queries']]
    print(f"Загружено {len(queries)} запросов.")

    try:
        vk_session = vk_api.VkApi(token=VK_TOKEN)
        vk = vk_session.get_api()
    except Exception as e:
        print(f"Ошибка авторизации ВК: {e}")
        return

    all_posts = []
    seen_post_links = set() # Общий кэш ссылок, чтобы не дублировать
    
    # Основной цикл по запросам
    for q_idx, query in enumerate(queries):
        print(f"\n🔎 [{q_idx+1}/{len(queries)}] Глобальный поиск: '{query}'")
        
        posts_collected_for_query = 0
        next_from = None # Маркер для пагинации (листания страниц поиска)

        # Цикл пагинации (пока не наберем MAX_POSTS_PER_QUERY или пока ВК не скажет хватит)
        while posts_collected_for_query < MAX_POSTS_PER_QUERY:
            try:
                # newsfeed.search ищет везде
                response = vk.newsfeed.search(
                    q=query, 
                    count=200, # Максимум за 1 раз
                    extended=1, # Чтобы получить инфу об авторах сразу
                    start_from=next_from
                )
                
                items = response.get('items', [])
                if not items:
                    break # Больше ничего нет по этому запросу

                for post in items:
                    # Фильтр по дате
                    post_date = datetime.fromtimestamp(post['date'])
                    if post_date < datetime.now() - timedelta(days=DAYS_TO_CHECK):
                        continue

                    # Достаем текст + текст репоста
                    text = post.get('text', '')
                    if 'copy_history' in post and len(post['copy_history']) > 0:
                        text += "\n--- REPOST ---\n" + post['copy_history'][0].get('text', '')
                    
                    if not text.strip(): continue

                    # Формируем ссылку и проверяем дубликаты
                    owner_id = post['owner_id']
                    post_id = post['id']
                    post_link = f"https://vk.com/wall{owner_id}_{post_id}"
                    
                    if post_link in seen_post_links:
                        continue
                    seen_post_links.add(post_link)

                    # Определени автора
                    # owner_id < 0 -> Группа
                    # owner_id > 0 -> Человек
                    author_type = "Человек" if owner_id > 0 else "Группа"
                    author_link = f"https://vk.com/id{owner_id}" if owner_id > 0 else f"https://vk.com/public{abs(owner_id)}"
                    
                    # Пытаемся достать красивое имя (из extended=1)
                    author_name = "?"
                    # (Тут можно было бы искать в response['profiles'] и ['groups'], но для простоты оставим "?",
                    # так как ссылка важнее. Ссылки достаточно, чтобы понять кто это)

                    all_posts.append({
                        'search_query': query,
                        'date': post_date.strftime('%Y-%m-%d'),
                        'author_type': author_type,
                        'author_link': author_link,
                        'phone': extract_phone(text),
                        'city': '?', 
                        'link': post_link,
                        'text': text[:5000]
                    })
                    
                    posts_collected_for_query += 1
                
                # Получаем код для следующей страницы
                next_from = response.get('next_from')
                if not next_from:
                    break # Страницы кончились
                
                time.sleep(1) # Пауза между страницами поиска

            except Exception as e:
                print(f"Ошибка при запросе: {e}")
                break

        print(f"   Собрано: {posts_collected_for_query} постов")

    # Сохранение
    print("\n")
    if all_posts:
        df = pd.DataFrame(all_posts)
        filename = f"global_search_{datetime.now().strftime('%m%d_%H%M')}.xlsx"
        df.to_excel(filename, index=False)
        print(f"Найдено {len(df)} записей по всему ВК.")
        print(f"Файл: {filename}")
    else:
        print("Ничего не найдено.")

if __name__ == "__main__":
    main()
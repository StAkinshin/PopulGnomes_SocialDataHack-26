"""
NLP-анализ объявлений о наборе в рабочие дома
"""

import re
import warnings
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'

# ═══════════════════════════════════════════════════════════════
# СТОП-СЛОВА (встроенные, без nltk)
# ═══════════════════════════════════════════════════════════════
STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
    'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
    'вы', 'за', 'бы', 'по', 'только', 'её', 'мне', 'было', 'вот', 'от',
    'меня', 'ещё', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
    'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был',
    'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там',
    'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
    'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам',
    'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
    'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем',
    'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы',
    'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно',
    'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над',
    'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них',
    'какая', 'много', 'разве', 'три', 'эту', 'впрочем', 'хорошо', 'свою',
    'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой',
    'более', 'всегда', 'конечно', 'всю', 'между', 'еще', 'это', 'который',
    'также', 'очень', 'просто', 'нужно', 'свой', 'весь', 'ваш', 'наш',
    'самый', 'каждый', 'другой', 'делать', 'мочь', 'будем', 'будет',
    'всё', 'вся', 'всем',
}


# ═══════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
        r'\U00002702-\U000027B0\U000024C2-\U0001F251'
        r'\U0000200d\U0000fe0f]+', ' ', text)
    text = re.sub(r'[^а-яёa-z0-9\s.,!?;:\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str, min_len: int = 3) -> List[str]:
    text = re.sub(r'[^а-яё\s]', ' ', clean_text(text))
    return [t for t in text.split() if len(t) >= min_len and t not in STOPWORDS]


def extract_salary(text: str) -> dict:
    text_lower = text.lower() if isinstance(text, str) else ""
    amounts = []
    patterns = [
        r'(\d[\d\s]*)\s*(?:руб|₽|рубл)',
        r'зп\s*(?:от\s*)?(\d[\d\s]*)',
        r'оплата\s*(?:от\s*)?(\d[\d\s]*)',
        r'(\d{3,})\s*(?:р[\.\s]|р\b)',
        r'от\s*(\d[\d\s]*)\s*(?:до\s*(\d[\d\s]*))?\s*(?:руб|₽|р\b)',
    ]
    for pat in patterns:
        for m in re.finditer(pat, text_lower):
            for g in m.groups():
                if g:
                    val = int(re.sub(r'\s', '', g))
                    if 100 <= val <= 500000:
                        amounts.append(val)
    return {'min': min(amounts) if amounts else None,
            'max': max(amounts) if amounts else None}


# ═══════════════════════════════════════════════════════════════
# ДОМЕННЫЕ СЛОВАРИ
# ═══════════════════════════════════════════════════════════════

OFFER_KEYWORDS = {
    'проживание', 'питание', 'жильё', 'жилье', 'общежитие',
    'бесплатно', 'бесплатное', 'обеспечим', 'предоставляем',
    'трехразовое', 'трёхразовое', 'документы', 'восстановление',
    'регистрация', 'трудоустройство', 'оформление', 'официально',
    'выплаты', 'ежедневно', 'ежедневная', 'стирка', 'постельное',
}

URGENCY_MARKERS = {'срочно', 'немедленно', 'сегодня', 'сейчас', 'быстро'}
MANIPULATION_MARKERS = {'уникальный', 'невероятный', 'гарантия', 'лучший', 'шанс'}
RISK_MARKERS = {'штраф', 'удержание', 'залог', 'изъятие', 'вычет', 'депозит', 'наличкой'}
VULNERABILITY_MARKERS = {
    'трудная', 'тяжелая', 'тяжёлая', 'сложная', 'ситуация', 'жизненная',
    'бездомн', 'попавш', 'помощь', 'помочь', 'поддержк', 'кризис',
    'беда', 'проблем', 'одинок', 'негде', 'некуда',
}


# ═══════════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС
# ═══════════════════════════════════════════════════════════════

class WorkhouseAnalyzer:

    def __init__(self, filepath: str = None):
        self.df = None
        self.results = {}
        if filepath:
            self.load_data(filepath)

    def load_data(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath)
        if path.suffix in ('.xlsx', '.xls'):
            self.df = pd.read_excel(filepath)
        elif path.suffix == '.csv':
            self.df = pd.read_csv(filepath, encoding='utf-8')
        else:
            raise ValueError(f"Формат {path.suffix} не поддерживается")

        for col in ['текст', 'Текст', 'text']:
            if col in self.df.columns:
                self.df = self.df.rename(columns={col: 'text'})
                break

        self.df = self.df.dropna(subset=['text'])
        self.df = self.df[self.df['text'].str.len() >= 20].reset_index(drop=True)
        self.df['text_clean'] = self.df['text'].apply(clean_text)
        self.df['tokens'] = self.df['text'].apply(tokenize)
        self.df['tokens_str'] = self.df['tokens'].apply(lambda x: ' '.join(x))
        self.df['char_len'] = self.df['text'].str.len()
        self.df['word_count'] = self.df['tokens'].apply(len)

        print(f"Загружено: {len(self.df)} объявлений, "
              f"средняя длина: {self.df['char_len'].mean():.0f} символов")
        return self.df

    # ── 1. Частотный анализ ────────────────────────────────────

    def frequency_analysis(self, top_n: int = 40) -> dict:
        texts = self.df['tokens_str'].tolist()
        all_tokens = [t for toks in self.df['tokens'] for t in toks]

        word_freq = pd.DataFrame(
            Counter(all_tokens).most_common(top_n), columns=['word', 'count'])
        word_freq['pct'] = (word_freq['count'] / len(all_tokens) * 100).round(2)

        bigram_vec = CountVectorizer(ngram_range=(2, 2), max_features=top_n,
                                     stop_words=list(STOPWORDS), min_df=2)
        bg = bigram_vec.fit_transform(texts)
        bigram_freq = pd.DataFrame({
            'bigram': bigram_vec.get_feature_names_out(),
            'count': bg.sum(axis=0).A1
        }).sort_values('count', ascending=False).reset_index(drop=True)

        trigram_vec = CountVectorizer(ngram_range=(3, 3), max_features=top_n,
                                      stop_words=list(STOPWORDS), min_df=2)
        tg = trigram_vec.fit_transform(texts)
        trigram_freq = pd.DataFrame({
            'trigram': trigram_vec.get_feature_names_out(),
            'count': tg.sum(axis=0).A1
        }).sort_values('count', ascending=False).reset_index(drop=True)

        tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=300,
                                     stop_words=list(STOPWORDS), min_df=2, max_df=0.95)
        tfidf_m = tfidf_vec.fit_transform(texts)
        tfidf_kw = pd.DataFrame({
            'keyword': tfidf_vec.get_feature_names_out(),
            'score': tfidf_m.mean(axis=0).A1
        }).sort_values('score', ascending=False).head(top_n).reset_index(drop=True)

        result = dict(word_freq=word_freq, bigram_freq=bigram_freq,
                      trigram_freq=trigram_freq, tfidf_keywords=tfidf_kw,
                      total_tokens=len(all_tokens), unique_tokens=len(set(all_tokens)))
        self.results['frequency'] = result
        print(f"Частотный анализ: {result['unique_tokens']} уникальных из {result['total_tokens']}")
        return result

    # ── 2. LDA ─────────────────────────────────────────────────

    def topic_modeling(self, n_topics: int = 6, top_words: int = 12) -> dict:
        texts = self.df['tokens_str'].tolist()
        vec = CountVectorizer(max_features=2000, ngram_range=(1, 2),
                              stop_words=list(STOPWORDS), min_df=2, max_df=0.92)
        dtm = vec.fit_transform(texts)
        feat = vec.get_feature_names_out()

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                         max_iter=25, learning_method='online')
        doc_topics = lda.fit_transform(dtm)

        topic_keywords = {}
        for i, comp in enumerate(lda.components_):
            idx = comp.argsort()[:-top_words - 1:-1]
            topic_keywords[i] = [feat[j] for j in idx]

        self.df['topic'] = doc_topics.argmax(axis=1)
        self.df['topic_confidence'] = doc_topics.max(axis=1)
        self.df['topic_label'] = self.df['topic'].map(
            {i: ', '.join(kw[:4]) for i, kw in topic_keywords.items()})

        result = dict(topic_keywords=topic_keywords, doc_topic_dist=doc_topics,
                      n_topics=n_topics)
        self.results['topics'] = result

        print(f"\nLDA: {n_topics} тем")
        for i, kw in topic_keywords.items():
            c = (self.df['topic'] == i).sum()
            print(f"  Тема {i} ({c} док.): {', '.join(kw[:6])}")
        return result

    # ── 3. Доменный анализ ─────────────────────────────────────

    def domain_analysis(self) -> dict:
        def count_m(text, markers):
            t = text.lower() if isinstance(text, str) else ''
            return sum(1 for m in markers if m in t)

        self.df['offer_score'] = self.df['text'].apply(lambda t: count_m(t, OFFER_KEYWORDS))
        self.df['risk_score'] = self.df['text'].apply(lambda t: count_m(t, RISK_MARKERS))
        self.df['urgency_score'] = self.df['text'].apply(lambda t: count_m(t, URGENCY_MARKERS))
        self.df['manipulation_score'] = self.df['text'].apply(lambda t: count_m(t, MANIPULATION_MARKERS))
        self.df['vulnerability_score'] = self.df['text'].apply(lambda t: count_m(t, VULNERABILITY_MARKERS))
        self.df['exclamation_count'] = self.df['text'].apply(
            lambda t: t.count('!') if isinstance(t, str) else 0)
        self.df['caps_ratio'] = self.df['text'].apply(
            lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1) if isinstance(t, str) else 0)

        sal = self.df['text'].apply(extract_salary)
        self.df['salary_extracted_min'] = sal.apply(lambda x: x['min'])
        self.df['salary_extracted_max'] = sal.apply(lambda x: x['max'])

        for label, kws in [
            ('mentions_housing', ['проживан', 'жильё', 'жилье', 'общежит', 'койко']),
            ('mentions_food', ['питани', 'кормим', 'трехразов', 'трёхразов', 'обед']),
            ('mentions_documents', ['документ', 'восстановлен', 'регистрац', 'оформлен']),
            ('mentions_daily_pay', ['ежедневн', 'каждый день']),
            ('mentions_free_exit', ['свободный выход', 'выход свободн', 'уйти в любой']),
        ]:
            self.df[label] = self.df['text_clean'].apply(lambda t: any(k in t for k in kws))

        binary_feats = {c: int(self.df[c].sum()) for c in
                        ['mentions_housing', 'mentions_food', 'mentions_documents',
                         'mentions_daily_pay', 'mentions_free_exit']}
        result = dict(binary_features=binary_feats,
                      salary_count=int(self.df['salary_extracted_min'].notna().sum()))
        self.results['domain'] = result

        print(f"\nДоменный анализ:")
        for f, cnt in binary_feats.items():
            print(f"  {f}: {cnt} ({cnt / len(self.df) * 100:.0f}%)")
        return result

    # ── 4. Клише ───────────────────────────────────────────────

    def cliche_analysis(self) -> dict:
        patterns = {
            'трудная жизненная ситуация': r'тр[ую]дн\w*\s+жизненн\w*\s+ситуаци',
            'проживание и питание': r'прожива\w*\s+и\s+пита\w*',
            'ежедневная оплата/выплата': r'ежедневн\w*\s+(оплат|выплат|зарплат)',
            'восстановление документов': r'восстановлен\w*\s+документ',
            'свободный выход': r'свободн\w*\s+выход|выход\s+свободн',
            'с документами и без': r'с\s+документ\w*\s+и\s+без',
            'бесплатное проживание': r'бесплатн\w*\s+прожива',
            'уютный дом/квартира': r'уютн\w*\s+(дом|кварт|жиль)',
            'трёхразовое питание': r'тр[её]х\s*разов\w*\s+питани',
            'от ... руб в день': r'от\s+\d+\s*(?:руб|р\b)',
            'работа для всех': r'работ\w*\s+для\s+всех',
            'попавших в ситуацию': r'попавш\w*\s+в\s+(?:\w+\s+)?ситуаци',
            'стирка и глажка': r'стирк\w*[\s,]+глажк',
            'чистая ухоженная квартира': r'чист\w*\s+(?:и\s+)?ухоженн',
            'обеспечим жильём': r'обеспечим\s+(?:вам\s+)?жиль',
            'постельное бельё': r'постельно\w*\s+бель',
            'без вредных привычек': r'без\s+вредн\w*\s+привыч',
        }

        cliche_counts = {}
        for name, pat in patterns.items():
            cnt = self.df['text_clean'].apply(lambda t: bool(re.search(pat, t))).sum()
            if cnt > 0:
                cliche_counts[name] = cnt

        cdf = pd.DataFrame([
            {'cliche': n, 'count': c, 'pct': round(c / len(self.df) * 100, 1)}
            for n, c in sorted(cliche_counts.items(), key=lambda x: -x[1])
        ])
        result = dict(cliche_df=cdf, cliche_counts=cliche_counts)
        self.results['cliches'] = result

        print(f"\nКлише ({len(cdf)} паттернов):")
        for _, r in cdf.head(8).iterrows():
            print(f"  «{r['cliche']}»: {r['count']} ({r['pct']}%)")
        return result

    # ── 5. Структурный анализ ─────────────────────────────────

    def structure_analysis(self) -> dict:
        self.df['has_numbered_list'] = self.df['text'].apply(
            lambda t: bool(re.search(r'\d+[\.\)]\s*\w', t)))
        self.df['has_bullet_points'] = self.df['text'].apply(
            lambda t: bool(re.search(r'[•\-✅✔️➡️▸]\s*\w', t)))
        self.df['has_emoji'] = self.df['text'].apply(
            lambda t: bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', t)))
        self.df['has_caps_words'] = self.df['text'].apply(
            lambda t: len(re.findall(r'\b[А-ЯЁA-Z]{4,}\b', t)) >= 3)

        feats = ['has_numbered_list', 'has_bullet_points', 'has_emoji', 'has_caps_words']
        stats = {f: int(self.df[f].sum()) for f in feats}
        result = dict(structure_stats=stats)
        self.results['structure'] = result

        print(f"\nСтруктура:")
        for f, c in stats.items():
            print(f"  {f}: {c} ({c / len(self.df) * 100:.0f}%)")
        return result

    # ── Запуск всего ──────────────────────────────────────────

    def run_all(self, n_topics: int = 6) -> dict:
        print("=" * 60)
        print("АНАЛИЗ ОБЪЯВЛЕНИЙ О НАБОРЕ В РАБОЧИЕ ДОМА")
        print("=" * 60)
        self.frequency_analysis()
        self.topic_modeling(n_topics=n_topics)
        self.domain_analysis()
        self.cliche_analysis()
        self.structure_analysis()
        print("\n" + "=" * 60)
        print("АНАЛИЗ ЗАВЕРШЁН")
        print("=" * 60)
        return self.results

    # ── Визуализации ──────────────────────────────────────────

    def save_visualizations(self, output_dir: str = 'results'):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []

        PAL = ['#2D5F8A', '#E8763A', '#59A14F', '#B07AA1',
               '#FF9DA7', '#9D7660', '#D4A017', '#4E79A7',
               '#F28E2B', '#76B7B2']

        sns.set_style("whitegrid")
        plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150,
                             'font.size': 11, 'axes.titlesize': 13})

        # 1 — Топ слов
        if 'frequency' in self.results:
            wf = self.results['frequency']['word_freq'].head(25)
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(wf)), wf['count'], color=PAL[0], alpha=0.85)
            ax.set_yticks(range(len(wf))); ax.set_yticklabels(wf['word'], fontsize=10)
            ax.invert_yaxis(); ax.set_xlabel('Частота')
            ax.set_title('Топ-25 слов в объявлениях', fontweight='bold')
            for bar, pct in zip(bars, wf['pct']):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f'{pct:.1f}%', va='center', fontsize=8, color='#555')
            plt.tight_layout(); fig.savefig(out / '01_top_words.png', bbox_inches='tight')
            plt.close(fig); saved.append('01_top_words.png')

        # 2 — Биграммы
        if 'frequency' in self.results:
            bf = self.results['frequency']['bigram_freq'].head(20)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(range(len(bf)), bf['count'], color=PAL[1], alpha=0.85)
            ax.set_yticks(range(len(bf))); ax.set_yticklabels(bf['bigram'], fontsize=10)
            ax.invert_yaxis(); ax.set_xlabel('Частота')
            ax.set_title('Топ-20 биграмм', fontweight='bold')
            plt.tight_layout(); fig.savefig(out / '02_top_bigrams.png', bbox_inches='tight')
            plt.close(fig); saved.append('02_top_bigrams.png')

        # 3 — Триграммы
        if 'frequency' in self.results:
            tf = self.results['frequency']['trigram_freq'].head(15)
            if len(tf) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(tf)), tf['count'], color=PAL[2], alpha=0.85)
                ax.set_yticks(range(len(tf))); ax.set_yticklabels(tf['trigram'], fontsize=10)
                ax.invert_yaxis(); ax.set_xlabel('Частота')
                ax.set_title('Топ-15 триграмм', fontweight='bold')
                plt.tight_layout(); fig.savefig(out / '03_top_trigrams.png', bbox_inches='tight')
                plt.close(fig); saved.append('03_top_trigrams.png')

        # 4 — TF-IDF
        if 'frequency' in self.results:
            ti = self.results['frequency']['tfidf_keywords'].head(20)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(range(len(ti)), ti['score'], color=PAL[3], alpha=0.85)
            ax.set_yticks(range(len(ti))); ax.set_yticklabels(ti['keyword'], fontsize=10)
            ax.invert_yaxis(); ax.set_xlabel('TF-IDF Score')
            ax.set_title('Ключевые слова по TF-IDF', fontweight='bold')
            plt.tight_layout(); fig.savefig(out / '04_tfidf.png', bbox_inches='tight')
            plt.close(fig); saved.append('04_tfidf.png')

        # 5 — Темы LDA
        if 'topics' in self.results:
            tk = self.results['topics']['topic_keywords']
            tc = self.df['topic'].value_counts().sort_index()
            fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 2]})
            cp = PAL[:len(tc)]
            axes[0].pie(tc.values, labels=[f'Тема {i}' for i in tc.index],
                        autopct='%1.0f%%', colors=cp, startangle=90)
            axes[0].set_title('Распределение по темам', fontweight='bold')
            labels = [f"Тема {i}: {', '.join(tk[i][:7])}" for i in sorted(tk)]
            axes[1].barh(list(range(len(tc))), tc.values, color=cp, alpha=0.85)
            axes[1].set_yticks(list(range(len(tc)))); axes[1].set_yticklabels(labels, fontsize=9)
            axes[1].set_xlabel('Кол-во объявлений')
            axes[1].set_title('Темы и ключевые слова', fontweight='bold')
            plt.tight_layout(); fig.savefig(out / '05_topics.png', bbox_inches='tight')
            plt.close(fig); saved.append('05_topics.png')

        # 6 — Клише
        if 'cliches' in self.results and len(self.results['cliches']['cliche_df']) > 0:
            cdf = self.results['cliches']['cliche_df']
            fig, ax = plt.subplots(figsize=(10, max(5, len(cdf) * 0.4)))
            ax.barh(range(len(cdf)), cdf['pct'], color=PAL[4], alpha=0.85)
            ax.set_yticks(range(len(cdf)))
            ax.set_yticklabels([f'«{c}»' for c in cdf['cliche']], fontsize=10)
            ax.invert_yaxis(); ax.set_xlabel('% объявлений')
            ax.set_title('Клише и шаблонные конструкции', fontweight='bold')
            for i, (_, row) in enumerate(cdf.iterrows()):
                ax.text(row['pct'] + 0.3, i, f"{row['count']}", va='center', fontsize=9, color='#555')
            plt.tight_layout(); fig.savefig(out / '06_cliches.png', bbox_inches='tight')
            plt.close(fig); saved.append('06_cliches.png')

        # 7 — Обещания (бинарные)
        if 'domain' in self.results:
            bf = self.results['domain']['binary_features']
            nm = {'mentions_housing': 'Проживание', 'mentions_food': 'Питание',
                  'mentions_documents': 'Документы', 'mentions_daily_pay': 'Ежедневная ЗП',
                  'mentions_free_exit': 'Свободный выход'}
            names = [nm.get(k, k) for k in bf]; vals = [v / len(self.df) * 100 for v in bf.values()]
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(names, vals, color=PAL[:len(names)], alpha=0.85)
            ax.set_ylabel('% объявлений'); ax.set_title('Что обещают в объявлениях', fontweight='bold')
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                        f'{v:.0f}%', ha='center', fontsize=10, fontweight='bold')
            plt.tight_layout(); fig.savefig(out / '07_offers.png', bbox_inches='tight')
            plt.close(fig); saved.append('07_offers.png')

        # 8 — Scatter: обещания vs уязвимость
        if 'offer_score' in self.df.columns:
            fig, ax = plt.subplots(figsize=(9, 7))
            sc = ax.scatter(self.df['offer_score'], self.df['vulnerability_score'],
                            c=self.df['exclamation_count'], cmap='YlOrRd',
                            s=self.df['caps_ratio'] * 500 + 20, alpha=0.6,
                            edgecolors='#333', linewidth=0.5)
            ax.set_xlabel('Маркеры предложений'); ax.set_ylabel('Маркеры уязвимости')
            ax.set_title('Обещания vs Апелляция к уязвимости\n(цвет = ! знаки, размер = CAPS)',
                         fontweight='bold')
            plt.colorbar(sc, label='Восклицательные знаки')
            plt.tight_layout(); fig.savefig(out / '08_scatter.png', bbox_inches='tight')
            plt.close(fig); saved.append('08_scatter.png')

        # 9 — Длины
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(self.df['char_len'], bins=30, color=PAL[0], alpha=0.75)
        axes[0].axvline(self.df['char_len'].median(), color='red', linestyle='--',
                        label=f"Медиана: {self.df['char_len'].median():.0f}")
        axes[0].set_xlabel('Символы'); axes[0].set_ylabel('Кол-во')
        axes[0].set_title('Длина текстов', fontweight='bold'); axes[0].legend()
        axes[1].hist(self.df['word_count'], bins=30, color=PAL[1], alpha=0.75)
        axes[1].axvline(self.df['word_count'].median(), color='red', linestyle='--',
                        label=f"Медиана: {self.df['word_count'].median():.0f}")
        axes[1].set_xlabel('Слов'); axes[1].set_ylabel('Кол-во')
        axes[1].set_title('Кол-во слов', fontweight='bold'); axes[1].legend()
        plt.tight_layout(); fig.savefig(out / '09_lengths.png', bbox_inches='tight')
        plt.close(fig); saved.append('09_lengths.png')

        # 10 — Структура
        if 'structure' in self.results:
            ss = self.results['structure']['structure_stats']
            nm = {'has_numbered_list': 'Нумерованный\nсписок', 'has_bullet_points': 'Буллеты',
                  'has_emoji': 'Эмодзи', 'has_caps_words': 'КАПС (≥3)'}
            names = [nm.get(k, k) for k in ss]; vals = [v / len(self.df) * 100 for v in ss.values()]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(names, vals, color=PAL[5], alpha=0.85)
            ax.set_ylabel('% объявлений'); ax.set_title('Форматирование объявлений', fontweight='bold')
            for b, v in zip(ax.patches, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                        f'{v:.0f}%', ha='center', fontsize=10)
            plt.tight_layout(); fig.savefig(out / '10_structure.png', bbox_inches='tight')
            plt.close(fig); saved.append('10_structure.png')

        # 11 — Зарплаты
        sal = self.df['salary_extracted_min'].dropna()
        if len(sal) > 5:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(sal, bins=25, color=PAL[6], alpha=0.75)
            ax.axvline(sal.median(), color='red', linestyle='--',
                       label=f"Медиана: {sal.median():.0f} руб.")
            ax.set_xlabel('Руб.'); ax.set_ylabel('Кол-во')
            ax.set_title('Зарплаты из текстов', fontweight='bold'); ax.legend()
            plt.tight_layout(); fig.savefig(out / '11_salary.png', bbox_inches='tight')
            plt.close(fig); saved.append('11_salary.png')

        # 12 — Сводный дашборд
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle('Сводный дашборд: объявления рабочих домов',
                     fontsize=18, fontweight='bold', y=0.98)

        ax1 = fig.add_subplot(gs[0, 0])
        wf10 = self.results['frequency']['word_freq'].head(10)
        ax1.barh(range(len(wf10)), wf10['count'], color=PAL[0], alpha=0.85)
        ax1.set_yticks(range(len(wf10))); ax1.set_yticklabels(wf10['word'], fontsize=9)
        ax1.invert_yaxis(); ax1.set_title('Топ-10 слов', fontweight='bold', fontsize=11)

        if 'topic' in self.df.columns:
            ax2 = fig.add_subplot(gs[0, 1])
            tc = self.df['topic'].value_counts().sort_index()
            ax2.pie(tc.values, labels=[f'T{i}' for i in tc.index],
                    autopct='%1.0f%%', colors=PAL[:len(tc)], textprops={'fontsize': 9})
            ax2.set_title('Темы (LDA)', fontweight='bold', fontsize=11)

        if 'domain' in self.results:
            ax3 = fig.add_subplot(gs[0, 2])
            bf = self.results['domain']['binary_features']
            sn = ['Жильё', 'Еда', 'Докум.', 'Дневная ЗП', 'Св. выход']
            vals = [v / len(self.df) * 100 for v in bf.values()]
            ax3.bar(sn, vals, color=PAL[:len(sn)], alpha=0.85); ax3.set_ylabel('%')
            ax3.set_title('Обещания', fontweight='bold', fontsize=11)
            plt.setp(ax3.get_xticklabels(), fontsize=8, rotation=20)

        if 'cliches' in self.results and len(self.results['cliches']['cliche_df']) > 0:
            ax4 = fig.add_subplot(gs[1, :2])
            cdf8 = self.results['cliches']['cliche_df'].head(8)
            ax4.barh(range(len(cdf8)), cdf8['pct'], color=PAL[4], alpha=0.85)
            ax4.set_yticks(range(len(cdf8)))
            ax4.set_yticklabels([f'«{c}»' for c in cdf8['cliche']], fontsize=9)
            ax4.invert_yaxis(); ax4.set_xlabel('%')
            ax4.set_title('Топ-8 клише', fontweight='bold', fontsize=11)

        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(self.df['char_len'], bins=20, color=PAL[0], alpha=0.7)
        ax5.set_xlabel('Символы'); ax5.set_title('Длина текстов', fontweight='bold', fontsize=11)

        if 'offer_score' in self.df.columns:
            ax6 = fig.add_subplot(gs[2, :2])
            sc = ax6.scatter(self.df['offer_score'], self.df['vulnerability_score'],
                             c=self.df['exclamation_count'], cmap='YlOrRd', s=40, alpha=0.5,
                             edgecolors='#555', linewidth=0.3)
            ax6.set_xlabel('Маркеры предложений'); ax6.set_ylabel('Маркеры уязвимости')
            ax6.set_title('Обещания vs Уязвимость', fontweight='bold', fontsize=11)
            plt.colorbar(sc, ax=ax6, label='!')

        if 'structure' in self.results:
            ax7 = fig.add_subplot(gs[2, 2])
            ss = self.results['structure']['structure_stats']
            sn2 = ['Списки', 'Буллеты', 'Эмодзи', 'КАПС']
            sv = [v / len(self.df) * 100 for v in ss.values()]
            ax7.bar(sn2, sv, color=PAL[5], alpha=0.85); ax7.set_ylabel('%')
            ax7.set_title('Форматирование', fontweight='bold', fontsize=11)

        fig.savefig(out / '12_dashboard.png', bbox_inches='tight')
        plt.close(fig); saved.append('12_dashboard.png')

        print(f"\nСохранено {len(saved)} визуализаций в {out}/")
        return saved

    def export_data(self, output_dir: str = 'results'):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cols = [c for c in self.df.columns if c != 'tokens']
        df_exp = self.df[cols].copy()
        if 'tokens_str' in df_exp.columns:
            df_exp = df_exp.rename(columns={'tokens_str': 'tokens'})
        df_exp.to_csv(out / 'analyzed_data.csv', index=False, encoding='utf-8-sig')
        if 'frequency' in self.results:
            for key in ('word_freq', 'bigram_freq', 'trigram_freq', 'tfidf_keywords'):
                self.results['frequency'][key].to_csv(
                    out / f'{key}.csv', index=False, encoding='utf-8-sig')
        if 'cliches' in self.results:
            self.results['cliches']['cliche_df'].to_csv(
                out / 'cliches.csv', index=False, encoding='utf-8-sig')
        print(f"Данные экспортированы в {out}/")


if __name__ == '__main__':
    analyzer = WorkhouseAnalyzer('test_data.xlsx')
    results = analyzer.run_all(n_topics=6)
    analyzer.save_visualizations('results')
    analyzer.export_data('results')
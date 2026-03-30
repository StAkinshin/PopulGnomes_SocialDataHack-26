# Workhouses study in the Russian Federation

Developed during **Social Data Hackathon 2026**  
Teams: **PopulGnomes** & **No name**

## Teams

PopulGnomes
- Konstantin
- Mark 
- Anna 
- Maria
- Stepan 

No name:
- Artem
- Ekaterina
- Anastasia
- Konstantin

# Avito Scraper

Scrapes job listings from Avito using undetected-chromedriver.

## Install

pip install -r requirements.txt

Chrome must be installed on the system.

## Usage

**Single URL:**
python cli.py -u "https://www.avito.ru/all/vakansii?q=рабочий+дом"

**All queries from input.json:**
python batch.py

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-u` | required | Avito URL |
| `-p` | 0 (all) | Number of pages |
| `-d` | `./` | Output directory |
| `-f` | `adverts.json` | Output filename |
| `--no-headless` | off | Show browser window |
| `--input` | `input.json` | Batch query file |
| `--output` | `./results` | Batch output dir |

## Output

JSON array of listings with: `advertId`, `title`, `description`, `url`, `price`, `author`, `address`.
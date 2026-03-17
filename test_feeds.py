import feedparser
import requests

GOV_SOURCES = [
    ("PIB",  "https://pib.gov.in/RssMain.aspx?ModID=6&Lang=1"),
    ("IMD",  "https://mausam.imd.gov.in/imd_latest/contents/rss-feed.php"),
    ("NDMA", "https://ndma.gov.in/RSS/ndma.xml"),
    ("WHO",  "https://www.who.int/rss-feeds/news-releases.xml"),
]

for name, url in GOV_SOURCES:
    try:
        # Try requests first
        r = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0"
        })
        print(f"{name}: HTTP {r.status_code} | {len(r.content)} bytes")
        
        # Try feedparser
        feed = feedparser.parse(url)
        print(f"  feedparser entries: {len(feed.entries)}")
        if feed.entries:
            print(f"  first title: {feed.entries[0].get('title','')[:60]}")
        if feed.bozo:
            print(f"  bozo error: {feed.bozo_exception}")
            
    except Exception as e:
        print(f"{name}: FAILED — {e}")
    print()
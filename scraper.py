import requests
from bs4 import BeautifulSoup

def extract_job_details(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        )
    }

    r = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    title = soup.title.text.strip() if soup.title else ""

    description = " ".join(
        tag.get_text(" ", strip=True)
        for tag in soup.find_all(["p", "li"])
    )[:5000]

    return title, description

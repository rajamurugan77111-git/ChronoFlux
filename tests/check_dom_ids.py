import re
from pathlib import Path

html = Path("frontend/index.html").read_text(encoding="utf-8")
js = Path("frontend/app.js").read_text(encoding="utf-8")
ids = set(re.findall(r"\$\('([^']+)'\)", js))
missing = sorted(i for i in ids if f'id="{i}"' not in html and f"id='{i}'" not in html)
# Dynamic page ids are constructed as page-${page}
dynamic_ok = {"pageTitle", "pageSubtitle", "sidebar"} <= ids
print("js_ids", len(ids))
print("missing", missing if missing else "none")
print("nav_pages", all(f'id="page-{p}"' in html for p in [
    "overview", "simulation", "live", "devices", "offline", "forecast", "alerts", "evidence", "system"
]))

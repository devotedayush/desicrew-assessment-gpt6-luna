"""Public definition lookup. No private workbook rows are accepted or sent."""
from __future__ import annotations
import html
import json
import re
import urllib.parse
import urllib.request

# Only public conceptual phrases can reach the web tool.
CONCEPTS = {
    "fifo": "FIFO inventory accounting",
    "lifo": "LIFO inventory accounting",
    "inventory": "inventory management",
    "stock": "inventory stock management",
    "sku": "stock keeping unit",
    "reorder": "reorder point",
    "stockout": "stockout",
    "turnover": "inventory turnover",
    "safety stock": "safety stock",
    "lead time": "lead time",
    "economic order": "economic order quantity",
    "abc": "ABC analysis",
    "valuation": "inventory valuation",
    "profit": "gross profit",
    "margin": "gross margin",
    "quantity": "inventory management",
}

class PublicDefinitionSearch:
    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self.last_error = None
        self.last_query = None

    def search(self, query: str) -> list[dict]:
        self.last_error = None
        lowered = str(query).lower()
        matches = [(k, v) for k, v in CONCEPTS.items() if k in lowered]
        specific = [(k, v) for k, v in matches if k not in {"inventory", "stock", "quantity"}]
        concept = (specific or matches)[0][1] if matches else None
        if not concept:
            self.last_error = "No public inventory concept matched; no query was transmitted"
            return []
        self.last_query = concept
        params = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": concept,
                                        "format": "json", "srlimit": 3, "srprop": "snippet"})
        request = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + params,
                                         headers={"User-Agent": "DesiCrewAssessment/0.1 (public definitions only)"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ValueError("Search response too large")
            data = json.loads(raw)
            return [{"title": row["title"],
                     "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(row["title"].replace(" ", "_")),
                     "snippet": html.unescape(re.sub(r"<[^>]+>", "", row.get("snippet", "")))}
                    for row in data["query"]["search"]]
        except Exception as exc:
            self.last_error = "Public definition search unavailable (" + type(exc).__name__ + ")"
            return []

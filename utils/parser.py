import re

def parse_lab_report(text):
    patterns = {
        "haemoglobin": r"haemoglobin\s*[:\-]?\s*([\d.]+)",
        "mcv": r"\bMCV\b\s*[:\-]?\s*([\d.]+)",
        "platelet count": r"platelet\s*count\s*[:\-]?\s*([\d.]+)",
        "t.l.c": r"(?:TLC|T\.L\.C|Total\s*Leukocyte\s*Count)\s*[:\-]?\s*([\d.]+)",
    }

    results = []
    for name, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            results.append({
                "name": name,
                "value": match.group(1),
                "unit": "",  # Optionally extract units with another pattern
            })

    return results

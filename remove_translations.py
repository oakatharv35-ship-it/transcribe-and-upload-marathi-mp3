import json
from pathlib import Path

with open(Path("gita_all_700_verses.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = {"verses": []}

for verse in data['verses']:
    new_verse = {
        "verse_id": verse["verse_uid"],
        "chapter_number": verse["chapter_number"],
        "verse_number": verse["verse_number"],
        "shloka_devanagari": verse["shloka_devanagari"],
        "shloka_iast": verse["shloka_iast"],
        "meaning": verse['translations'][0]['meaning']

    }
    new_data["verses"].append(new_verse)

with open(Path("gita_translated.json"), "w", encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
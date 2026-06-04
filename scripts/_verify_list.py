from pathlib import Path
import re

path = Path("data/lists/us_tech_list.md")
text = path.read_text(encoding="utf-8")
lines = text.splitlines()

for l in lines[:5]:
    print(l)
print("...")
hc = [l for l in lines if "| Healthcare |" in l]
print(f"Healthcare 行残留: {len(hc)}")
data_rows = [l for l in lines if re.match(r"^\| \d+", l)]
print(f"数据行总数: {len(data_rows)}")
print("最后一行:", data_rows[-1][:80])

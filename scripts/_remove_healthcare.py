"""从 us_tech_list.md 中删除 Healthcare 行，重新编号，更新头部。"""
from pathlib import Path
import re
from datetime import date

path = Path("data/lists/us_tech_list.md")
text = path.read_text(encoding="utf-8")
lines = text.splitlines()

# 统计
hc_lines = [l for l in lines if "| Healthcare |" in l]
print(f"Healthcare 行数: {len(hc_lines)}")

# 保留非 Healthcare 数据行（及表头、分隔符、注释）
out_lines = []
new_num = 0
for line in lines:
    # 是表格数据行（以 | 数字 开头）
    if re.match(r"^\|\s*\d+\s*\|", line):
        if "| Healthcare |" in line:
            continue  # 删除
        new_num += 1
        # 替换行首编号
        line = re.sub(r"^\|\s*\d+\s*\|", f"| {new_num} |", line)
    # 更新头部总数注释
    elif re.match(r"^> 共\s*\d+\s*只", line):
        line = re.sub(r"\d+", str(new_num), line, count=1)
    # 更新最后更新日期
    elif re.match(r"^> 自动生成，最后更新", line):
        line = re.sub(r"\d{4}-\d{2}-\d{2}", date.today().isoformat(), line)
    out_lines.append(line)

new_text = "\n".join(out_lines) + "\n"
path.write_text(new_text, encoding="utf-8")
print(f"完成: {new_num} 只保留，{len(hc_lines)} 只 Healthcare 已删除")
print(f"已写入: {path}")

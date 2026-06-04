import sys, json
from pathlib import Path
sys.path.insert(0, 'src')

out_dir = Path('data/output/smc_ob_scan')
files = sorted(out_dir.glob('*_futu_events.json'))
if files:
    f = files[-1]
    payload = json.loads(f.read_text(encoding='utf-8'))
    print(f"最新事件文件: {f.name}")
    print(f"  date={payload['date']}  total={payload['total']}")
    print(f"  new_ob={len(payload['new_ob'])}  mitigated={len(payload['mitigated'])}  touched={len(payload['touched'])}")
else:
    print("无事件文件")

state_dir = Path('data/cache/smc_ob_state')
markets = [d.name for d in state_dir.iterdir() if d.is_dir()]
print(f"状态文件目录: {markets}")
for m in markets:
    n = len(list((state_dir/m).glob('*.json')))
    print(f"  {m}: {n} 个状态文件")

"""Generate wave-structure charts for selected stocks."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_ana.data.market_data import load_watchlist_data
from stock_ana.strategies.primitives.wave import analyze_wave_structure
from stock_ana.utils.plot_renderers import plot_wave_structure_chart

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "output" / "wave_structure_charts"

TARGETS = ["00981", "NVDA", "PLTR", "000537", "AAPL", "PDD", "MSFT"]


def main():
    data = load_watchlist_data()
    for sym in TARGETS:
        info = data.get(sym)
        if info is None:
            print(f"[SKIP] {sym} not found in shawn data")
            continue
        df = info["df"]
        market = info["market"]
        name = info["name"]
        ws = analyze_wave_structure(df)
        waves = ws["major_waves"]
        print(f"{market}:{sym} {name}  waves={len(waves)}  status={ws['current_status']}")
        for w in waves:
            ep_val = w["end_pivot"]["value"] if w["end_pivot"] else "ongoing"
            print(
                f"  W{w['wave_number']}: "
                f"start@{w['start_pivot']['value']:.2f} "
                f"peak@{w['peak_pivot']['value']:.2f} "
                f"end@{ep_val}  "
                f"+{w['rise_pct']:.0f}%  subs={w['sub_wave_count']}"
            )
        p = plot_wave_structure_chart(sym, market, name, df, waves, OUT_DIR)
        if p:
            print(f"  -> {p}")
        else:
            print("  -> (no chart generated)")


if __name__ == "__main__":
    main()

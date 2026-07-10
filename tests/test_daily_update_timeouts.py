import json
import threading

import pandas as pd

import daily_update
from stock_ana.data import fetcher


class _FakeRequests:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return object()


requests = _FakeRequests()


def _fake_stock_us_daily(symbol: str, adjust: str) -> pd.DataFrame:
    requests.get(f"https://example.test/{symbol}")
    return pd.DataFrame({"symbol": [symbol], "adjust": [adjust]})


def test_akshare_call_injects_and_restores_request_timeout():
    backend = requests
    backend.calls.clear()

    result = fetcher._call_akshare_us_daily_with_timeout(
        _fake_stock_us_daily,
        symbol="TEST",
        adjust="qfq",
        timeout=(3.0, 7.0),
    )

    assert result.iloc[0]["symbol"] == "TEST"
    assert backend.calls == [{"url": "https://example.test/TEST", "timeout": (3.0, 7.0)}]
    assert _fake_stock_us_daily.__globals__["requests"] is backend


def test_akshare_stage_budget_stops_remaining_tickers(monkeypatch):
    now = {"value": 0.0}
    stale = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=[pd.Timestamp.now().normalize() - pd.Timedelta(days=5)],
    )

    def fake_fetch(symbol: str, start_date: str | None = None) -> pd.DataFrame:
        now["value"] = 2.0
        return pd.DataFrame()

    monkeypatch.setattr(fetcher.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(fetcher, "_fetch_us_stock_akshare", fake_fetch)

    result = fetcher._update_bucket_data(
        ["AAA", "BBB"],
        load_fn=lambda ticker: stale,
        save_fn=lambda ticker, frame: None,
        market_label="TEST",
        max_stale_days=1,
        stage_timeout_sec=1.0,
    )

    assert result["updated"] == 0
    assert result["failed"] == 1
    assert result["budget_exhausted"] is True


def test_watchdog_triggers_hard_exit_and_releases_lock(tmp_path, monkeypatch):
    status_calls: list[int] = []
    exit_codes: list[int] = []
    lock_path = tmp_path / "daily_update.lock"
    lock_path.write_text(f"pid={daily_update.os.getpid()}\n", encoding="utf-8")
    monkeypatch.setattr(daily_update, "LOCK_PATH", lock_path)
    monkeypatch.setattr(daily_update, "_write_watchdog_status", status_calls.append)

    daily_update._watchdog_main(
        threading.Event(),
        timeout_sec=0,
        exit_fn=exit_codes.append,
    )

    assert status_calls == [0]
    assert exit_codes == [daily_update.WATCHDOG_EXIT_CODE]
    assert not lock_path.exists()


def test_watchdog_status_preserves_scan_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_update, "PROJECT_ROOT", tmp_path)
    status_dir = tmp_path / "data" / "output" / "daily_update" / pd.Timestamp.now().date().isoformat()
    status_dir.mkdir(parents=True)
    status_path = status_dir / "status.json"
    status_path.write_text(
        json.dumps({"scan_ready": True, "in_progress": True, "steps": [{"name": "US", "ok": True}]}),
        encoding="utf-8",
    )

    daily_update._write_watchdog_status(120)
    status = json.loads(status_path.read_text(encoding="utf-8"))

    assert status["scan_ready"] is True
    assert status["in_progress"] is False
    assert status["all_ok"] is False
    assert status["timed_out"] is True
    assert status["steps"] == [{"name": "US", "ok": True}]

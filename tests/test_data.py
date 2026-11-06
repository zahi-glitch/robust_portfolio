"""The new data tools, tested without internet: yfinance and Wikipedia are replaced by small fakes."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

import robustfolio.data as data
from robustfolio.data import (_parse_sp500_changes, balance, coverage, download_prices, history, members_as_of,
                              _members_from_tables, _yahoo_symbol)

DAYS = pd.bdate_range("2020-01-01", periods=60)


def fake_prices():
    p = pd.DataFrame({"OLD": 100.0 + np.arange(60), "MID": 50.0 + np.arange(60), "NEW": 10.0 + np.arange(60)}, index=DAYS)
    p.loc[DAYS[:20], "MID"] = np.nan       # starts on day 20
    p.loc[DAYS[:40], "NEW"] = np.nan       # starts on day 40
    p.index.name = "date"
    return p


def test_balance_all_assets_short_sample():
    out, dropped = balance(fake_prices())
    assert list(out.columns) == ["OLD", "MID", "NEW"] and out.index[0] == DAYS[40] and dropped == []


def test_balance_long_sample_fewer_assets():
    out, dropped = balance(fake_prices(), start=DAYS[0], tolerance_days=0)
    assert list(out.columns) == ["OLD"] and len(out) == 60 and dropped == ["MID", "NEW"]
    out, dropped = balance(fake_prices(), start=DAYS[20], tolerance_days=0)
    assert list(out.columns) == ["OLD", "MID"] and out.index[0] == DAYS[20] and dropped == ["NEW"]


def test_balance_drops_assets_that_stopped_trading():
    p = fake_prices()
    p.loc[DAYS[50]:, "OLD"] = np.nan
    out, dropped = balance(p, start=DAYS[20], tolerance_days=0)
    assert "OLD" not in out.columns and "OLD" in dropped


def test_history_is_sorted_youngest_first():
    h = history(fake_prices())
    assert list(h.index) == ["NEW", "MID", "OLD"] and h.loc["MID", "days"] == 40


def test_symbols():
    assert _yahoo_symbol(" brk.b ") == "BRK-B"


CURRENT = ["AAA", "BBB", "NEW1", "NEW2"]
CHANGES = pd.DataFrame({
    "date": pd.to_datetime(["2010-06-01", "2015-03-01", "2020-09-01"]),
    "added": ["BBB", "NEW1", "NEW2"],
    "removed": ["OLD1", "AAA2", float("nan")],       # the last change only added a company
})


def test_members_as_of_undoes_changes():
    assert members_as_of("2021-01-01", CURRENT, CHANGES) == sorted(CURRENT)
    assert members_as_of("2016-01-01", CURRENT, CHANGES) == ["AAA", "BBB", "NEW1"]
    assert members_as_of("2012-01-01", CURRENT, CHANGES) == ["AAA", "AAA2", "BBB"]
    assert members_as_of("2000-01-01", CURRENT, CHANGES) == ["AAA", "AAA2", "OLD1"]


def test_parse_wikipedia_tables():
    cols = pd.MultiIndex.from_tuples([("Effective Date", "Effective Date"), ("Added", "Ticker"),
                                      ("Added", "Security"), ("Removed", "Ticker"), ("Removed", "Security")])
    t = pd.DataFrame([["September 1, 2020", "NEW2", "x", None, None], ["March 1, 2015", "NEW1", "y", "AAA.2", "z"]],
                     columns=cols)
    ch = _parse_sp500_changes([pd.DataFrame({"a": [1]}), t])
    assert list(ch["added"]) == ["NEW2", "NEW1"] and ch["removed"].isna().tolist() == [True, False] and ch["removed"].iloc[1] == "AAA-2"
    assert ch["date"].iloc[0] == pd.Timestamp("2020-09-01")
    members = pd.DataFrame({"Symbol": [f"S{i}" for i in range(450)] + ["BRK.B"]})
    assert _members_from_tables([t, members], "Symbol", (400, 600))[-1] == "BRK-B"


def test_coverage():
    c = coverage(["OLD", "GONE", "MID"], fake_prices())
    assert c["with prices"] == 2 and c["missing"] == ["GONE"]


@pytest.fixture
def fake_yf(monkeypatch, tmp_path):
    calls = []
    mod = types.ModuleType("yfinance")

    def download(tickers, **kw):
        calls.append(list(tickers))
        known = [t for t in tickers if t != "BAD"]        # Yahoo knows every ticker but BAD
        close = fake_prices().assign(**{t: 1.0 for t in known if t not in ("OLD", "MID", "NEW")})
        close = close[[t for t in known if t in close.columns]]
        return pd.concat({"Close": close}, axis=1) if len(close.columns) > 1 else pd.concat({"Close": close}, axis=1)

    mod.download = download
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    monkeypatch.setattr(data, "CACHE", tmp_path)
    return calls


def test_download_chunks_cache_and_missing(fake_yf):
    with pytest.warns(UserWarning, match="BAD"):
        df = download_prices(["OLD", "MID", "NEW", "BAD"], chunk=2, balanced=False)
    assert list(df.columns) == ["OLD", "MID", "NEW"] and len(fake_yf) == 2        # two chunks of two
    assert df["NEW"].isna().sum() == 40                                             # unbalanced kept
    again = download_prices(["OLD", "MID", "NEW"], chunk=2, balanced=False)         # different key -> new download
    n = len(fake_yf)
    with pytest.warns(UserWarning):
        download_prices(["OLD", "MID", "NEW", "BAD"], chunk=2, balanced=False)       # same key -> from cache
    assert len(fake_yf) == n and again.shape == df.shape
    assert len(download_prices(["OLD", "MID", "NEW"], chunk=2, balanced=True)) == 20  # balanced: shortest history

"""The new data tools, tested without internet: yfinance and Wikipedia are replaced by small fakes."""
import io
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


CHANGES_HTML = """
<table><thead>
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr></thead>
<tbody>
<tr><td>September 1, 2020</td><td>NEW2</td><td>x</td><td></td><td></td><td>growth</td></tr>
<tr><td>March 1, 2015</td><td>NEW1[1]</td><td>y</td><td>AAA.2</td><td>z</td><td>merger</td></tr>
</tbody></table>"""
PAD = "<table><tr><th>a</th></tr><tr><td>1</td></tr></table>"


def test_parse_wikipedia_changes_table():
    """A real HTML table with spanning headers, read both ways pandas can (header inferred, or header=None)."""
    for kw in ({}, {"header": None}):
        ch = _parse_sp500_changes(pd.read_html(io.StringIO(PAD + CHANGES_HTML), **kw))
        check_changes(ch)
    flat = pd.DataFrame([["Effective Date", "Added", "Added", "Removed", "Removed", "Reason"],     # header as plain rows
                         ["Effective Date", "Ticker", "Security", "Ticker", "Security", "Reason"],
                         ["September 1, 2020", "NEW2", "x", None, None, "growth"],
                         ["March 1, 2015", "NEW1[1]", "y", "AAA.2", "z", "merger"]])
    check_changes(_parse_sp500_changes([PAD_DF, flat]))


PAD_DF = pd.DataFrame({"a": [1]})


def check_changes(ch):
    assert ch["added"].tolist() == ["NEW2", "NEW1"] and ch["added"].notna().all()
    assert ch["removed"].isna().tolist() == [True, False] and ch["removed"].iloc[1] == "AAA-2"
    assert ch["date"].iloc[0] == pd.Timestamp("2020-09-01")
    assert members_as_of("2016-01-01", ["NEW1", "NEW2", "AAA"], ch) == ["AAA", "NEW1"]


def test_parse_failure_says_what_it_saw():
    big = pd.DataFrame([[f"r{i}", 1, 2, 3] for i in range(60)])
    with pytest.raises(RuntimeError, match="Large tables seen"):
        _parse_sp500_changes([big])


def test_members_table():
    members = pd.DataFrame({"Symbol": [f"S{i}" for i in range(450)] + ["BRK.B"]})
    assert _members_from_tables([pd.DataFrame({"a": [1]}), members], "Symbol", (400, 600))[-1] == "BRK-B"


def test_balance_start_before_any_data_is_an_error():
    with pytest.raises(ValueError, match="earliest start"):
        balance(fake_prices(), start="2019-01-01")


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

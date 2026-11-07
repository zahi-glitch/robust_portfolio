"""Prices and returns.

Sources
-------
  load_prices      the bundled file: daily prices of US stocks, 1990-2018 (works offline)
  download_prices  Yahoo Finance via yfinance (needs internet and `pip install -e ".[live]"`)
  load_universe    a named universe ("etfs", "sectors", "dow30", "nasdaq100", "sp500", "sp500@2005-01-03"),
                   downloaded once per day and cached on disk

All return a pandas DataFrame: one row per trading day, one column per ticker.

Bigger and less biased samples
------------------------------
Three ways to get more assets than the 10 bundled stocks, with different biases:

  ETFS      ten liquid ETFs, one per asset class (from 2006). A fund is not "a company that survived": there
            is no survivorship bias at the asset level, but the sample is short.
  SECTORS   the nine original SPDR sector ETFs (from Dec 1998). The longest history, and it covers the whole
            US market, including the 2000-02 and 2008 crises.
  members   the stocks of an index. Today's members are the winners (survivorship bias): `sp500_members_as_of`
            rebuilds the list as it was on a past date from Wikipedia's table of index changes. Yahoo has
            no prices for most companies that were delisted since, so some of the old members are missing:
            `coverage` says how many, i.e. how much bias is left.

Assets start on different dates. `download_prices` can return the unbalanced panel (NaN before an asset
exists); `balance` turns it into the rectangular table the backtest needs, and tells you which assets it
had to drop to keep a long sample (more assets <-> a longer sample: `history` shows the trade-off).
"""
from __future__ import annotations

import hashlib
import io
import re
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

PRICES = Path(__file__).with_name("data") / "stock_prices.csv"
CACHE = Path(__file__).resolve().parents[2] / "data_cache"      # next to pyproject.toml; git-ignored
# The 10 stocks of the bundled file with a full history from 1990 to 2018.
STOCKS = ["AAPL", "GE", "AMD", "WMT", "BAC", "T", "XOM", "BBY", "PFE", "JPM"]

ETFS = {
    "SPY": "US stocks (S&P 500)",
    "EFA": "stocks, other developed countries",
    "EEM": "stocks, emerging markets",
    "VNQ": "US real estate",
    "TLT": "long-term US government bonds",
    "IEF": "medium-term US government bonds",
    "SHY": "short-term US government bonds (near cash)",
    "LQD": "US corporate bonds",
    "GLD": "gold",
    "DBC": "commodities (from Feb 2006)",
}
SECTORS = {
    "XLB": "materials", "XLE": "energy", "XLF": "financials", "XLI": "industrials", "XLK": "technology",
    "XLP": "consumer staples", "XLU": "utilities", "XLV": "health care", "XLY": "consumer discretionary",
}                                                                       # all trade since December 1998
SECTORS_NEW = {"XLRE": "real estate (from 2015)", "XLC": "communication services (from 2018)"}

UNIVERSES = {"etfs": list(ETFS), "sectors": list(SECTORS), "sectors_all": list(SECTORS) + list(SECTORS_NEW)}
_WIKI = "https://en.wikipedia.org/wiki/"
_INDEX_PAGES = {                     # name -> (page, column holding the ticker, plausible number of members)
    "sp500": ("List_of_S%26P_500_companies", "Symbol", (400, 600)),
    "dow30": ("Dow_Jones_Industrial_Average", "Symbol", (28, 32)),
    "nasdaq100": ("Nasdaq-100", "Ticker", (90, 110)),
}


# --------------------------------------------------------------------------- bundled file
def load_prices(tickers=STOCKS, start=None, end=None, path=None) -> pd.DataFrame:
    """Daily adjusted closing prices from the bundled CSV (or `path`)."""
    df = pd.read_csv(path or PRICES, parse_dates=[0], index_col=0).sort_index()
    df.index.name = "date"
    if tickers is not None:
        df = df[list(tickers)]
    return df.loc[start:end].dropna()


# --------------------------------------------------------------------------- Yahoo Finance
def _yahoo_symbol(s) -> str:
    """Wikipedia writes BRK.B, Yahoo wants BRK-B."""
    return str(s).strip().upper().replace(".", "-")


def _download_chunk(yf, tickers, start, end) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if data is None or len(data) == 0:
        return pd.DataFrame()
    close = data["Close"]
    if isinstance(close, pd.Series):                    # a single ticker comes back as a Series
        close = close.to_frame(tickers[0])
    return close


def download_prices(tickers, start="2005-01-01", end=None, balanced=True, chunk=50, cache=True) -> pd.DataFrame:
    """Daily closing prices adjusted for dividends and splits, from Yahoo Finance.

    balanced=True   only the days on which *every* ticker has a price (the sample is as short as the youngest
                    asset: the behaviour of the first version).
    balanced=False  the unbalanced panel: NaN before an asset exists or after it stops trading. Pass it to
                    `balance` (or look at `history`) to choose between more assets and a longer sample.

    Tickers are fetched `chunk` at a time. The result is cached in `data_cache/` for the rest of the day
    (when `end` is None), so re-running a notebook does not download again. Tickers that Yahoo does not know
    are reported in a warning and left out.
    """
    tickers = list(dict.fromkeys(_yahoo_symbol(t) for t in tickers))
    key = hashlib.md5(f"{sorted(tickers)}|{start}|{end}|{date.today() if end is None else ''}".encode()).hexdigest()[:12]
    file = CACHE / f"prices_{key}.csv"
    if cache and file.exists():
        df = pd.read_csv(file, parse_dates=[0], index_col=0)
    else:
        try:
            import yfinance as yf
        except ImportError as e:
            raise RuntimeError('yfinance is not installed: pip install -e ".[live]"') from e
        parts = [_download_chunk(yf, tickers[i:i + chunk], start, end) for i in range(0, len(tickers), chunk)]
        parts = [p for p in parts if len(p)]
        if not parts:
            raise RuntimeError("no prices downloaded: check the internet connection")
        df = pd.concat(parts, axis=1)
        df = df.loc[:, ~df.columns.duplicated()].dropna(how="all", axis=1).dropna(how="all").sort_index()
        df.index.name = "date"
        if cache:
            CACHE.mkdir(exist_ok=True)
            df.to_csv(file)
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        warnings.warn(f"no prices for {len(missing)} of {len(tickers)} tickers: {', '.join(missing[:15])}"
                      + (" ..." if len(missing) > 15 else ""), stacklevel=2)
    df = df[[t for t in tickers if t in df.columns]].ffill(limit=2)       # bridge one-off holiday mismatches
    return df.dropna() if balanced else df


def history(prices: pd.DataFrame) -> pd.DataFrame:
    """First and last day, and number of prices, of each asset, youngest first."""
    h = pd.DataFrame({"first": prices.apply(pd.Series.first_valid_index),
                      "last": prices.apply(pd.Series.last_valid_index),
                      "days": prices.notna().sum()})
    return h.sort_values("first", ascending=False)


def balance(prices: pd.DataFrame, start=None, tolerance_days=7):
    """From an unbalanced panel to a rectangular table. Returns (prices, dropped tickers).

    start=None   keep every asset, and only the days on which all of them exist.
    start=date   keep the assets that already trade on `start` (give or take `tolerance_days` calendar days)
                 and the days from `start` on: a long sample, with fewer assets.
    Assets that stopped trading before the end of the panel are dropped in both cases.
    """
    full = prices.dropna(how="all")
    alive = full.columns[full.iloc[-3:].notna().any()]                    # still quoted at the end
    first = full[alive].apply(pd.Series.first_valid_index)
    if start is None:
        keep = list(alive)
    else:
        start = pd.Timestamp(start)
        keep = [t for t in alive if first[t] <= start + pd.Timedelta(days=tolerance_days)]
        if not keep:
            raise ValueError(f"no asset has prices on {start.date()}: the earliest start is {first.min().date()}")
        full = full.loc[start:]
    out = full[keep].dropna()
    return out, [t for t in prices.columns if t not in keep]


# --------------------------------------------------------------------------- index members (Wikipedia)
def _wiki_html(page) -> str:
    from urllib.request import Request, urlopen
    req = Request(_WIKI + page, headers={"User-Agent": "robustfolio (educational project)"})
    return urlopen(req, timeout=30).read().decode("utf-8")


def _read_tables(html, **kw) -> list:
    try:
        return pd.read_html(io.StringIO(html), **kw)
    except ImportError as e:
        raise RuntimeError('pandas.read_html needs lxml: pip install -e ".[live]"') from e


def _wiki_tables(page) -> list:
    return _read_tables(_wiki_html(page))


def _members_from_tables(tables, column, size) -> list:
    for t in tables:
        t = t.copy()
        t.columns = [" ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in t.columns]
        if column in t.columns and size[0] <= len(t) <= size[1]:
            return [_yahoo_symbol(s) for s in t[column].dropna()]
    raise RuntimeError(f"no table with {size[0]}-{size[1]} rows and a '{column}' column: Wikipedia changed?")


def index_members(index="sp500") -> list:
    """Today's members of "sp500", "dow30" or "nasdaq100", as Yahoo tickers (survivorship bias: see above)."""
    page, column, size = _INDEX_PAGES[index]
    return _members_from_tables(_wiki_tables(page), column, size)


def _clean_ticker(s):
    """'BRK.B' -> 'BRK-B', drops footnote marks like 'ABC[1]'; empty -> None."""
    if pd.isna(s):
        return None
    s = re.sub(r"\[.*?\]", "", str(s)).strip()
    return _yahoo_symbol(s) if s and s.lower() != "nan" else None


def _as_grid(t: pd.DataFrame) -> pd.DataFrame:
    """A table as a plain grid of cells: header rows (one per header level, if any) on top of the body."""
    if isinstance(t.columns, pd.MultiIndex):
        head = [list(t.columns.get_level_values(i)) for i in range(t.columns.nlevels)]
    elif isinstance(t.columns, pd.RangeIndex):
        head = []
    else:
        head = [list(t.columns)]
    body = [list(r) for r in t.astype(object).to_numpy()]
    return pd.DataFrame(head + body, dtype=object)


def _parse_sp500_changes(raw_tables) -> pd.DataFrame:
    """Wikipedia's table of S&P 500 changes -> DataFrame(date, added, removed).

    Two-row headers (a spanning 'Added' above 'Ticker' and 'Security') may come as MultiIndex columns, as
    plain header rows, or as flat labels, depending on the page and the pandas version: `_as_grid` makes
    them all rows, and the Added/Removed rows are searched in the first three.
    """
    seen = []
    for t in raw_tables:
        t = _as_grid(t)
        g = t.astype(object).where(t.notna(), "").astype(str).apply(lambda col: col.str.strip().str.lower())
        for i in range(min(3, len(g))):                               # which of the first rows holds Added/Removed?
            row = list(g.iloc[i])
            if any(c.startswith("add") for c in row) and any(c.startswith("remov") for c in row):
                sub = list(g.iloc[i + 1]) if i + 1 < len(g) else row
                has_sub = any("ticker" in c or "symbol" in c for c in sub)

                def find(group):
                    for c, (r, u) in enumerate(zip(row, sub)):
                        if r.startswith(group) and (not has_sub or "ticker" in u or "symbol" in u):
                            return c
                    raise RuntimeError(f"no ticker column under '{group}'")

                when = next(c for c, r in enumerate(row) if "date" in r)
                add, rem = find("add"), find("remov")
                body = t.iloc[i + (2 if has_sub else 1):]
                out = pd.DataFrame({"date": pd.to_datetime(body.iloc[:, when].astype(str).str.replace(r"\[.*?\]", "", regex=True), errors="coerce"),
                                    "added": body.iloc[:, add].map(_clean_ticker),
                                    "removed": body.iloc[:, rem].map(_clean_ticker)})
                out = out.dropna(subset=["date"]).reset_index(drop=True)
                if len(out):
                    return out
        if len(t) > 50 and t.shape[1] >= 4:
            seen.append(f"  {t.shape}: " + " | ".join(map(str, t.iloc[0].tolist()[:6])))
    raise RuntimeError("no table of changes found: Wikipedia changed? Large tables seen (shape: first row):\n"
                       + "\n".join(seen[:6]))


def members_as_of(when, current, changes) -> list:
    """Members on `when`, by undoing, newest first, every change made after that date."""
    members = set(current)
    for _, row in changes[changes["date"] > pd.Timestamp(when)].sort_values("date", ascending=False).iterrows():
        if pd.notna(row["added"]):
            members.discard(row["added"])
        if pd.notna(row["removed"]):
            members.add(row["removed"])
    return sorted(members)


def sp500_members_as_of(when) -> list:
    """The S&P 500 as it was on `when`, rebuilt from Wikipedia (accurate to its table of changes, ~1990 on)."""
    html = _wiki_html(_INDEX_PAGES["sp500"][0])
    current = _members_from_tables(_read_tables(html), "Symbol", _INDEX_PAGES["sp500"][2])
    return members_as_of(when, current, _parse_sp500_changes(_read_tables(html)))


def coverage(wanted, prices: pd.DataFrame) -> dict:
    """How much of a list of tickers has prices? The rest (mostly delisted companies) is the bias left."""
    have = [t for t in wanted if t in prices.columns]
    return {"wanted": len(wanted), "with prices": len(have), "share": len(have) / max(len(wanted), 1),
            "missing": [t for t in wanted if t not in prices.columns]}


# --------------------------------------------------------------------------- named universes
def load_universe(name, start="2005-01-01", end=None, balanced=False) -> pd.DataFrame:
    """Prices of a named universe, downloaded (and cached for the day).

    name  "etfs", "sectors", "sectors_all", "dow30", "nasdaq100", "sp500" (today's members), or
          "sp500@YYYY-MM-DD" (the members on that date; look at `coverage` for what is missing).
    Returns the unbalanced panel unless balanced=True; see `balance` and `history`.
    """
    if name in UNIVERSES:
        tickers = UNIVERSES[name]
    elif name in _INDEX_PAGES:
        tickers = index_members(name)
    elif name.startswith("sp500@"):
        tickers = sp500_members_as_of(name.split("@", 1)[1])
    else:
        raise ValueError(f"unknown universe {name!r}: {sorted([*UNIVERSES, *_INDEX_PAGES])} or 'sp500@YYYY-MM-DD'")
    return download_prices(tickers, start=start, end=end, balanced=balanced)


def returns_pct(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns in %: row t is the return from day t-1 to day t."""
    return 100.0 * prices.pct_change().iloc[1:]

"""Download a universe once, look at its history, and save a balanced table.

    pip install -e ".[live]"
    python scripts/fetch_data.py sectors                       # 9 SPDR sector ETFs, since 1998
    python scripts/fetch_data.py etfs --start 2005-01-01
    python scripts/fetch_data.py sp500@2005-01-03 --balance-from 2005-01-03
    python scripts/fetch_data.py dow30 --balance-from 2000-01-03 --out data_cache/dow30_2000.csv

Prints, for each asset, when its prices start, and how many assets a given start date keeps: the trade-off
between more assets and a longer sample. With --balance-from the rectangular table is written to --out.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from robustfolio import balance, history, load_universe
from robustfolio.data import _INDEX_PAGES, index_members, sp500_members_as_of, coverage

CACHE = Path(__file__).resolve().parents[1] / "data_cache"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("universe", help="etfs, sectors, sectors_all, dow30, nasdaq100, sp500, sp500@YYYY-MM-DD")
    ap.add_argument("--start", default="1990-01-01", help="first day to download (default 1990-01-01)")
    ap.add_argument("--balance-from", help="keep the assets that exist on this date, and write the table")
    ap.add_argument("--out", help="CSV to write (default data_cache/<universe>.csv)")
    a = ap.parse_args()

    prices = load_universe(a.universe, start=a.start)
    h = history(prices)
    print(f"{prices.shape[1]} assets with prices, {prices.index[0].date()} to {prices.index[-1].date()}\n")

    if a.universe.startswith("sp500@") or a.universe in _INDEX_PAGES:       # how much bias is left?
        wanted = sp500_members_as_of(a.universe.split("@")[1]) if "@" in a.universe else index_members(a.universe)
        c = coverage(wanted, prices)
        print(f"coverage: {c['with prices']} of {c['wanted']} members have prices ({100 * c['share']:.0f}%); "
              f"the others, mostly delisted, are missing: that is the survivorship bias left\n")

    print("youngest assets:\n", h.head(8).to_string(), "\n")
    print("assets kept if the sample starts at (latest start of each year; later years keep more assets)...")
    firsts = h["first"].sort_values()
    for y in sorted({d.year for d in firsts}):
        cutoff = firsts[firsts.dt.year == y].max()
        kept, _ = balance(prices, start=cutoff, tolerance_days=0)
        print(f"  {cutoff.date()}: {kept.shape[1]:4d} assets, {len(kept):5d} days")

    if a.balance_from:
        kept, dropped = balance(prices, start=a.balance_from)
        out = Path(a.out or CACHE / f"{a.universe.replace('@', '_')}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        kept.to_csv(out)
        print(f"\nwrote {out}: {kept.shape[1]} assets, {len(kept)} days; dropped {len(dropped)}: {dropped[:20]}")


if __name__ == "__main__":
    main()

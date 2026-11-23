"""Download and cache daily OHLCV for the optional market-data mode (yfinance).

    python scripts/download_market_data.py

Writes data/market/market_us.csv (+ _universe, _benchmark, .meta.json).  On any download
failure the script stops with an error: it NEVER substitutes synthetic data.

Conventions recorded in the metadata:
* provider: Yahoo Finance via yfinance (unofficial API; terms of use apply; the cached
  files are for local research and should not be redistributed);
* prices: ``auto_adjust=True`` -> OHLC adjusted for splits AND dividends (a total-return
  research approximation).  Share counts are therefore in adjusted units and the volume
  column is as returned by the provider; this is NOT an execution-grade share ledger;
* universe: 32 current US large caps in 7 industries, labels assigned by the author from
  current public GICS-style classifications on the retrieval date.  This is NOT a
  point-in-time universe: survivorship and look-ahead in membership are present.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    exp = yaml.safe_load((root / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    m = exp["market"]
    uni = pd.read_csv(root / m["universe_file"])
    if uni["currency"].nunique() != 1:
        print("mixed currencies in universe file: refusing", file=sys.stderr)
        return 1
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed (pip install yfinance)", file=sys.stderr)
        return 1
    syms = uni["symbol"].tolist()
    retrieved = datetime.now(UTC).isoformat(timespec="seconds")
    # yfinance 'end' is exclusive
    end_excl = (pd.Timestamp(m["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(syms + [m["benchmark"]], start=m["start"], end=end_excl,
                      auto_adjust=True, progress=False, group_by="ticker", threads=True)
    if raw is None or raw.empty:
        print("download failed or returned nothing; no data written", file=sys.stderr)
        return 2
    rows, missing = [], []
    for s in syms:
        if s not in raw.columns.get_level_values(0):
            missing.append(s)
            continue
        d = raw[s].rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        d = d.dropna(how="any")
        d = d[(d["close"] > 0) & (d["volume"] >= 0)]
        # repair tiny OHLC inconsistencies created by adjustment rounding (logged)
        hi = d[["open", "close", "high"]].max(axis=1)
        lo = d[["open", "close", "low"]].min(axis=1)
        d = d.assign(high=hi, low=lo)
        d.insert(0, "symbol", s)
        d.index = pd.to_datetime(d.index).tz_localize(None)
        d.index.name = "date"
        rows.append(d.reset_index())
    if missing:
        print(f"symbols missing from download: {missing}", file=sys.stderr)
    if not rows:
        return 2
    out = root / "data" / "market"
    oh = pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"])
    oh.to_csv(out / "market_us.csv", index=False, float_format="%.6f")
    uni[uni["symbol"].isin(oh["symbol"].unique())].to_csv(out / "market_us_universe.csv",
                                                          index=False)
    b = raw[m["benchmark"]]["Close"].dropna()
    b.index = pd.to_datetime(b.index).tz_localize(None)
    b.rename(m["benchmark"]).to_frame().rename_axis("date").to_csv(
        out / "market_us_benchmark.csv", float_format="%.6f")
    meta = {
        "data_mode": "market", "provider": "Yahoo Finance via yfinance",
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "retrieved_utc": retrieved, "requested_start": m["start"], "requested_end": m["end"],
        "actual_start": str(oh["date"].min().date()), "actual_end": str(oh["date"].max().date()),
        "symbols_requested": syms, "symbols_missing": missing,
        "benchmark": m["benchmark"], "currency": "USD",
        "price_adjustment": "auto_adjust=True: OHLC split- and dividend-adjusted",
        "volume": "as returned by provider (adjusted basis not guaranteed)",
        "missing_data_rule": "rows with any missing OHLCV dropped; never forward-filled",
        "ohlc_repair": "high/low widened to contain adjusted open/close where rounding broke it",
        "classification": "author-assigned current GICS-style industry labels on retrieval "
                          "date; historical changes NOT available",
        "point_in_time_membership": False,
        "survivorship_bias": "present: only companies listed on the retrieval date",
        "redistribution": "local research cache; do not redistribute raw provider data",
    }
    (out / "market_us.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(oh):,} rows for {oh['symbol'].nunique()} symbols "
          f"{meta['actual_start']}..{meta['actual_end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

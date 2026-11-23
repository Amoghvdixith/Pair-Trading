"""Create and EXECUTE the three notebooks (all calculations call package functions).

    python scripts/build_notebooks.py

Requires results from scripts/run_all.py for the tables the notebooks display.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SETUP = """\
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
TAB, FIG = ROOT / "results" / "tables", ROOT / "results" / "figures"
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
print("SYNTHETIC results demonstrate software behaviour; they are not market evidence.")"""

NB = {
    "01_screening_and_fdr.ipynb": [
        ("md", "# 01 - Screening and false-discovery control\n\nCorrelation is not "
               "cointegration; residual-based tests need MacKinnon critical values; BH "
               "controls the expected FALSE-DISCOVERY PROPORTION within one formation-date "
               "family, under independence/PRDS - not across repeated screens."),
        ("code", SETUP),
        ("code", "from pairs import data as D\nfrom pairs.screening import engle_granger, "
                 "johansen, benjamini_hochberg, benjamini_yekutieli\n"
                 "A, _ = D.cointegrated_pair(500, seed=1)\nC, _ = D.correlated_random_walks("
                 "500, rho=0.9, seed=1)\nfor name, df in [('cointegrated', A), ('correlated "
                 "RWs', C)]:\n    eg = engle_granger(df['y'], df['x'])\n    print(f\"{name:14s}"
                 " return corr={df['y'].diff().corr(df['x'].diff()):.2f}  EG stat={eg.stat:.2f}"
                 "  5% crit={eg.crit_5:.2f}  p={eg.pvalue:.3g}  Johansen rank="
                 "{johansen(df['y'], df['x'])['johansen_rank']}\")"),
        ("code", "p = [0.01, 0.04, 0.03, 0.20]\nprint('BH', benjamini_hochberg(p, 0.05))\n"
                 "print('BY', benjamini_yekutieli(p, 0.05))"),
        ("code", "display(pd.read_csv(TAB / 'syn_00_calibration.csv'))"),
        ("code", "display(Image(FIG / 'syn_02_screening_funnel.png'))\n"
                 "display(pd.read_csv(TAB / 'syn_02b_screening_vs_truth.csv').describe().T"
                 "[['mean']].T)"),
        ("code", "cand = pd.read_csv(TAB / 'syn_03_candidates_first_test_cohort.csv')\n"
                 "display(cand.head(12))"),
        ("code", "mk = TAB / 'mkt_02_screening_funnel_by_cohort.csv'\nif mk.exists():\n"
                 "    f = pd.read_csv(mk); print('MARKET: cohorts', len(f), 'with no selection',"
                 " int((f.selected == 0).sum()))\n    display(f[['candidates','raw_reject',"
                 "'fdr_reject','selected']].sum())"),
    ],
    "02_ou_and_kalman.ipynb": [
        ("md", "# 02 - OU half-life and Kalman hedge ratios\n\nHalf-life = ln2/theta with "
               "theta = -ln(b)/dt, valid only for 0<b<1.  The Kalman innovation is computed "
               "from the PREDICTED state; the posterior residual uses y_t and is not a "
               "tradable z-score."),
        ("code", SETUP),
        ("code", "from pairs import data as D\nfrom pairs.ou import fit_ou, hurst_exponent, "
                 "variance_ratio_test\nu = D.simulate_ar1(2000, 10.0, 0.05, "
                 "np.random.default_rng(0))\nf = fit_ou(u)\nprint(f.status, round(f.b, 4), "
                 "'half-life', round(f.half_life, 2), 'CI', tuple(round(v, 1) for v in "
                 "f.half_life_ci))\nprint('Hurst', hurst_exponent(u), '\\nVR', "
                 "variance_ratio_test(u, 5))\nprint('negative b ->', fit_ou(-0.5*np.ones(50)"
                 " + np.random.default_rng(1).normal(0, 1, 50)).status)"),
        ("code", "from pairs.hedge import tune_kalman, kalman_filter, pykalman_reference\n"
                 "df, _ = D.time_varying_beta_pair(1500, seed=3)\nt = tune_kalman(df.y[:400],"
                 " df.x[:400], [1e-7, 1e-6, 1e-5, 1e-4])\nkf, ll = kalman_filter(df.y, df.x, "
                 "t.Q, t.R, np.array(t.m0), np.diag(t.P0_diag))\nref = pykalman_reference("
                 "df.y, df.x, t.Q, t.R, np.array(t.m0), np.diag(t.P0_diag))\nprint('delta', "
                 "t.delta, 'max |ours - pykalman|', None if ref is None else "
                 "float(np.abs(ref.values - kf[['alpha','beta']].values).max()))\n"
                 "ax = df.beta_true.plot(label='true'); kf.beta.plot(ax=ax, label='Kalman'); "
                 "ax.legend(); plt.show()"),
        ("code", "display(Image(FIG / 'syn_05b_kalman_time_varying_beta.png'))\n"
                 "display(Image(FIG / 'syn_06_ou_fit_half_life.png'))"),
        ("code", "display(pd.read_csv(TAB / 'syn_14b_oos_spread_adf_by_method.csv')"
                 ".groupby('method').adf_p_nominal.median())\nprint('Adaptive (rolling/"
                 "Kalman) residuals look more stationary by construction: nominal ADF "
                 "p-values are not valid inference for them.')"),
    ],
    "03_portfolio_backtest_sensitivity.ipynb": [
        ("md", "# 03 - Portfolio backtest, costs and sensitivity\n\nTrades go through the "
               "project-02 backtester (signal at close, fill next open).  Gross = no costs, "
               "borrow or interest; net = all of them."),
        ("code", SETUP),
        ("code", "from pairs.experiments import build_context, run, summary\nfrom pairs "
                 "import analysis as A\nctx = build_context(ROOT, 'synthetic', 'fast')\n"
                 "plans = ctx.plans('test')[:2]  # two cohorts only, to keep this quick\n"
                 "rows = {}\nfor sc, bps in [('flat', 0.0), ('flat', 20.0)]:\n"
                 "    res, s = run(ctx, 'test', plans=plans, scenario=sc, flat_bps=bps)\n"
                 "    rows[f'{bps:g}bps'] = summary(ctx, res, s, sc)\n"
                 "pd.DataFrame(rows).loc[['sharpe','cagr','total_trading_costs',"
                 "'completed_pair_trades']]"),
        ("code", "pnl, rec = A.attribute_pairs(res, s, ctx.market_data('test'))\nprint("
                 "'max |engine - pairs - financing|:', float(rec.unattributed.abs().max()))"),
        ("code", "for f in ['syn_10_portfolio_metrics.csv', 'syn_12_cost_sensitivity.csv', "
                 "'syn_14_hedge_method_comparison.csv', 'mkt_10_portfolio_metrics.csv', "
                 "'mkt_14_hedge_method_comparison.csv']:\n    p = TAB / f\n    if p.exists():"
                 "\n        print(f); display(pd.read_csv(p).head(12))"),
        ("code", "for f in ['syn_13_threshold_heatmap_validation.png', "
                 "'syn_13b_threshold_heatmap_EXPLORATORY_test.png', "
                 "'mkt_08_equity_gross_net.png']:\n    display(Image(FIG / f))"),
    ],
}


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in NB.items():
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_markdown_cell(c) if k == "md" else
                    nbformat.v4.new_code_cell(c) for k, c in cells]
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(out)}}).execute()
        nbformat.write(nb, out / name)
        print(f"executed {name}")


if __name__ == "__main__":
    main()

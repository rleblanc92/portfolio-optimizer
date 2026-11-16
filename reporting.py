"""Self-contained institutional dashboard for out-of-sample backtests."""

from __future__ import annotations

from html import escape
from io import StringIO
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


STRATEGY_COLOR = "#1f4e78"
BENCHMARK_COLOR = "#b8860b"
SECONDARY_COLOR = "#888888"


def _drawdown_statistics(returns: np.ndarray) -> tuple[float, int, np.ndarray]:
    nav = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peaks = np.maximum.accumulate(nav)
    drawdown = nav / peaks - 1.0
    max_drawdown = float(np.min(drawdown))

    current_duration = 0
    max_duration = 0
    for value in drawdown[1:]:
        if value < 0.0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return max_drawdown, max_duration, nav[1:]


def _metrics(
    returns: np.ndarray,
    risk_free_rate: float,
    *,
    weights_history=None,
    turnover_history=None,
    transaction_cost_drag_bps_yr=None,
    margin_financing_drag_bps_yr=None,
) -> dict[str, float | int | None]:
    n_days = len(returns)
    cumulative_return = float(np.prod(1.0 + returns) - 1.0) if n_days else 0.0
    cagr = (
        float((1.0 + cumulative_return) ** (252.0 / n_days) - 1.0)
        if n_days and cumulative_return > -1.0 else -1.0
    )
    volatility = float(np.std(returns) * np.sqrt(252.0)) if n_days else 0.0
    daily_rf = risk_free_rate / 252.0
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 1e-12 else 0.0
    downside = np.minimum(returns - daily_rf, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside ** 2) * 252.0)) if n_days else 0.0
    sortino = (cagr - risk_free_rate) / downside_deviation if downside_deviation > 1e-12 else 0.0

    mean_return = float(np.mean(returns)) if n_days else 0.0
    centered = returns - mean_return
    variance = float(np.mean(centered ** 2)) if n_days else 0.0
    std_dev = np.sqrt(variance)
    if std_dev > 1e-12:
        skewness = float(np.mean(centered ** 3) / std_dev ** 3)
        excess_kurtosis = float(np.mean(centered ** 4) / std_dev ** 4 - 3.0)
    else:
        skewness = 0.0
        excess_kurtosis = 0.0

    max_drawdown, max_drawdown_duration, nav = _drawdown_statistics(returns)
    calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 1e-12 else 0.0
    if n_days:
        var_cutoff = float(np.quantile(returns, 0.05))
        tail_returns = returns[returns <= var_cutoff]
        var_95 = max(0.0, -var_cutoff)
        cvar_95 = max(0.0, -float(np.mean(tail_returns))) if len(tail_returns) else var_95
    else:
        var_95 = 0.0
        cvar_95 = 0.0

    beta = None
    alpha = None
    gross_exposure = None
    net_exposure = None
    if weights_history is not None:
        weight_array = np.asarray(weights_history, dtype=np.float64)
        if weight_array.ndim == 2 and weight_array.shape[0] > 0:
            gross_exposure = float(np.mean(np.sum(np.abs(weight_array), axis=1)))
            net_exposure = float(np.mean(np.sum(weight_array, axis=1)))

    turnover = None
    if turnover_history is not None and len(turnover_history):
        turnover = float(np.mean(np.asarray(turnover_history, dtype=np.float64)))

    return {
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "daily_var_95": var_95,
        "daily_cvar_95": cvar_95,
        "beta": beta,
        "annualized_alpha": alpha,
        "average_rebalance_turnover": turnover,
        "annualized_transaction_cost_drag_bps": transaction_cost_drag_bps_yr,
        "annualized_margin_financing_drag_bps": margin_financing_drag_bps_yr,
        "average_gross_exposure": gross_exposure,
        "average_net_exposure": net_exposure,
        "nav": nav,
    }


def _comparison_metrics(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float,
    strategy_metrics: dict,
    benchmark_metrics: dict,
) -> None:
    if len(strategy_returns) < 2 or len(benchmark_returns) != len(strategy_returns):
        return
    benchmark_variance = float(np.var(benchmark_returns))
    if benchmark_variance <= 1e-15:
        return
    beta = float(np.cov(strategy_returns, benchmark_returns, ddof=0)[0, 1] / benchmark_variance)
    daily_rf = risk_free_rate / 252.0
    alpha = float((np.mean(strategy_returns - daily_rf) - beta * np.mean(benchmark_returns - daily_rf)) * 252.0)
    strategy_metrics["beta"] = beta
    strategy_metrics["annualized_alpha"] = alpha


def _style_axis(axis) -> None:
    axis.grid(True, color="#eeeeee", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#cccccc")
    axis.spines["bottom"].set_color("#cccccc")


def _render_svg(figure, index: int) -> str:
    buffer = StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    svg = buffer.getvalue()
    svg = svg[svg.find("<svg"):]
    prefix = f"dashboard{index}-"
    svg = re.sub(r'\bid="([^"]+)"', lambda match: f'id="{prefix}{match.group(1)}"', svg)
    svg = re.sub(r'url\(#([^\)]+)\)', lambda match: f'url(#{prefix}{match.group(1)})', svg)
    svg = re.sub(r'(?:xlink:)?href="#([^"]+)"', lambda match: f'href="#{prefix}{match.group(1)}"', svg)
    return svg


def _format_percent(value) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.2%}"


def _format_number(value, digits: int = 2) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _metric_table(
    title: str,
    rows: list[tuple[str, str, str]],
    benchmark_name: str,
) -> str:
    row_html = "".join(
        f"<tr><th>{escape(label)}</th><td>{escape(strategy)}</td><td>{escape(benchmark)}</td></tr>"
        for label, strategy, benchmark in rows
    )
    return (
        f'<section class="metric-section"><h3>{escape(title)}</h3>'
        f'<table><thead><tr><th>Metric</th><th>Strategy</th><th>{escape(benchmark_name)}</th></tr></thead>'
        f"<tbody>{row_html}</tbody></table></section>"
    )


def _monthly_returns(returns: np.ndarray, dates: pd.DatetimeIndex) -> tuple[np.ndarray, list[int]]:
    if len(returns) == 0:
        return np.empty((0, 13)), []
    series = pd.Series(returns, index=dates)
    monthly = series.groupby(series.index.to_period("M")).apply(lambda values: np.prod(1.0 + values) - 1.0)
    years = sorted({period.year for period in monthly.index})
    matrix = np.full((len(years), 13), np.nan, dtype=np.float64)
    year_rows = {year: row for row, year in enumerate(years)}
    for period, value in monthly.items():
        matrix[year_rows[period.year], period.month - 1] = value
    annual = series.groupby(series.index.year).apply(lambda values: np.prod(1.0 + values) - 1.0)
    for year, value in annual.items():
        matrix[year_rows[year], 12] = value
    return matrix, years


def generate_institutional_report(
    strategy_returns,
    benchmark_returns,
    dates,
    *,
    scenario_name="Portfolio",
    benchmark_name="Benchmark",
    risk_free_rate=0.0,
    output_path="outputs/tearsheets/best_strategy_tearsheet.html",
    weights_history=None,
    rebalance_dates=None,
    turnover_history=None,
    transaction_costs=None,
    transaction_cost_drag_bps_yr=None,
    margin_financing_drag_bps_yr=None,
) -> Path:
    """Write a self-contained HTML dashboard with four inline SVG charts."""
    strategy_returns = np.asarray(strategy_returns, dtype=np.float64).reshape(-1)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64).reshape(-1)
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) != len(dates):
        raise ValueError("Strategy returns, benchmark returns and dates must have equal lengths")
    if not np.all(np.isfinite(strategy_returns)) or not np.all(np.isfinite(benchmark_returns)):
        raise ValueError("Dashboard returns must contain only finite values")

    strategy_metrics = _metrics(
        strategy_returns,
        risk_free_rate,
        weights_history=weights_history,
        turnover_history=turnover_history,
        transaction_cost_drag_bps_yr=transaction_cost_drag_bps_yr,
        margin_financing_drag_bps_yr=margin_financing_drag_bps_yr,
    )
    benchmark_metrics = _metrics(benchmark_returns, risk_free_rate)
    _comparison_metrics(strategy_returns, benchmark_returns, risk_free_rate, strategy_metrics, benchmark_metrics)

    turnover_values = np.asarray(turnover_history, dtype=np.float64) if turnover_history is not None else np.empty(0)
    turnover_dates = pd.DatetimeIndex(pd.to_datetime(rebalance_dates)) if rebalance_dates is not None else pd.DatetimeIndex([])
    transaction_cost_values = np.asarray(transaction_costs, dtype=np.float64) if transaction_costs is not None else np.empty(0)
    if len(turnover_values) != len(turnover_dates):
        usable = min(len(turnover_values), len(turnover_dates))
        turnover_values = turnover_values[:usable]
        turnover_dates = turnover_dates[:usable]
    if len(transaction_cost_values) != len(turnover_dates):
        transaction_cost_values = np.empty(0)

    svg_charts = []
    figures = []
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": "#eeeeee",
        "grid.linewidth": 0.8,
        "svg.fonttype": "none",
    }):
        try:
            figure, axes = plt.subplots(
                3,
                1,
                figsize=(12, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1.2, 1.2]},
            )
            figures.append(figure)
            strategy_nav = strategy_metrics["nav"]
            benchmark_nav = benchmark_metrics["nav"]
            axes[0].plot(dates, 100.0 * strategy_nav, color=STRATEGY_COLOR, linewidth=1.8, label="Strategy · net")
            axes[0].plot(dates, 100.0 * benchmark_nav, color=BENCHMARK_COLOR, linewidth=1.4, label=benchmark_name)
            axes[0].set_title("Cumulative Performance · Base 100", loc="left", weight="bold")
            axes[0].set_ylabel("Value")
            axes[0].legend(frameon=False, loc="upper left", ncol=2)

            strategy_drawdown = strategy_nav / np.maximum.accumulate(strategy_nav) - 1.0
            benchmark_drawdown = benchmark_nav / np.maximum.accumulate(benchmark_nav) - 1.0
            axes[1].fill_between(dates, strategy_drawdown, 0.0, color=STRATEGY_COLOR, alpha=0.18)
            axes[1].plot(dates, strategy_drawdown, color=STRATEGY_COLOR, linewidth=1.2, label="Strategy")
            axes[1].fill_between(dates, benchmark_drawdown, 0.0, color=BENCHMARK_COLOR, alpha=0.14)
            axes[1].plot(dates, benchmark_drawdown, color=BENCHMARK_COLOR, linewidth=1.0, label=benchmark_name)
            axes[1].set_title("Underwater · Drawdown", loc="left", weight="bold")
            axes[1].set_ylabel("Drawdown")
            axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))

            axes[2].set_title("Execution Turnover & Cost Drag", loc="left", weight="bold")
            axes[2].set_ylabel("L1 turnover")
            if len(turnover_values):
                axes[2].bar(turnover_dates, turnover_values, width=8, color=STRATEGY_COLOR, alpha=0.75, label="L1 turnover")
                average_turnover = float(np.mean(turnover_values))
                axes[2].axhline(average_turnover, color=SECONDARY_COLOR, linestyle="--", linewidth=1.2, label=f"Mean {average_turnover:.1%}")
                if len(transaction_cost_values):
                    cost_axis = axes[2].twinx()
                    cost_axis.plot(turnover_dates, transaction_cost_values * 10000.0, color=BENCHMARK_COLOR, marker="o", markersize=3, linewidth=1.0, label="Transaction cost · bps")
                    cost_axis.set_ylabel("Cost (bps)")
                    cost_axis.spines["top"].set_visible(False)
                    handles, labels = axes[2].get_legend_handles_labels()
                    cost_handles, cost_labels = cost_axis.get_legend_handles_labels()
                    axes[2].legend(handles + cost_handles, labels + cost_labels, frameon=False, ncol=3, loc="upper left")
                else:
                    axes[2].legend(frameon=False, loc="upper left")
            else:
                axes[2].text(0.5, 0.5, "Rebalance history unavailable", ha="center", va="center", transform=axes[2].transAxes, color=SECONDARY_COLOR)
            axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
            for axis in axes:
                _style_axis(axis)
            figure.tight_layout()
            svg_charts.append(_render_svg(figure, 1))

            figure, axis = plt.subplots(figsize=(11, 4.4))
            figures.append(figure)
            combined = np.concatenate((strategy_returns, benchmark_returns))
            bins = min(40, max(8, int(np.sqrt(max(len(combined), 1)))))
            axis.hist(strategy_returns, bins=bins, density=True, alpha=0.28, color=STRATEGY_COLOR, label="Strategy")
            axis.hist(benchmark_returns, bins=bins, density=True, alpha=0.22, color=BENCHMARK_COLOR, label=benchmark_name)
            if len(strategy_returns) > 2 and np.std(strategy_returns) > 1e-12:
                grid = np.linspace(float(np.min(combined)), float(np.max(combined)), 300)
                try:
                    axis.plot(grid, gaussian_kde(strategy_returns)(grid), color=STRATEGY_COLOR, linewidth=1.6)
                except (ValueError, np.linalg.LinAlgError):
                    pass
                if np.std(benchmark_returns) > 1e-12:
                    try:
                        axis.plot(grid, gaussian_kde(benchmark_returns)(grid), color=BENCHMARK_COLOR, linewidth=1.4)
                    except (ValueError, np.linalg.LinAlgError):
                        pass
            strategy_var = -strategy_metrics["daily_var_95"]
            strategy_cvar = -strategy_metrics["daily_cvar_95"]
            axis.axvline(strategy_var, color=STRATEGY_COLOR, linestyle="--", linewidth=1.2, label="Strategy VaR 95%")
            axis.axvline(strategy_cvar, color=STRATEGY_COLOR, linestyle=":", linewidth=1.5, label="Strategy CVaR 95%")
            axis.text(
                0.02,
                0.96,
                f"Skewness  {strategy_metrics['skewness']:.2f}\nExcess Kurtosis  {strategy_metrics['excess_kurtosis']:.2f}",
                transform=axis.transAxes,
                va="top",
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.95},
            )
            axis.set_title("Return Distribution & Tail Risk", loc="left", weight="bold")
            axis.set_xlabel("Daily return")
            axis.set_ylabel("Density")
            axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.1%}"))
            axis.legend(frameon=False, ncol=2, loc="upper right")
            _style_axis(axis)
            figure.tight_layout()
            svg_charts.append(_render_svg(figure, 2))

            figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            figures.append(figure)
            strategy_series = pd.Series(strategy_returns, index=dates)
            benchmark_series = pd.Series(benchmark_returns, index=dates)
            rolling_window = 126
            strategy_volatility = strategy_series.rolling(rolling_window, min_periods=rolling_window).std() * np.sqrt(252.0)
            benchmark_volatility = benchmark_series.rolling(rolling_window, min_periods=rolling_window).std() * np.sqrt(252.0)
            axes[0].plot(dates, strategy_volatility, color=STRATEGY_COLOR, label="Strategy")
            axes[0].plot(dates, benchmark_volatility, color=BENCHMARK_COLOR, label=benchmark_name)
            axes[0].set_title("Rolling 6-Month Annualized Volatility", loc="left", weight="bold")
            axes[0].set_ylabel("Volatility")
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
            axes[0].legend(frameon=False, ncol=2)
            rolling_covariance = strategy_series.rolling(rolling_window, min_periods=rolling_window).cov(benchmark_series)
            rolling_variance = benchmark_series.rolling(rolling_window, min_periods=rolling_window).var()
            rolling_beta = rolling_covariance / rolling_variance.replace(0.0, np.nan)
            axes[1].plot(dates, rolling_beta, color=STRATEGY_COLOR, label="Rolling beta")
            axes[1].axhline(1.0, color=SECONDARY_COLOR, linestyle="--", linewidth=1.0)
            axes[1].set_title("Rolling 6-Month Beta", loc="left", weight="bold")
            axes[1].set_ylabel("Beta")
            axes[1].set_xlabel("Date")
            for axis in axes:
                _style_axis(axis)
            figure.tight_layout()
            svg_charts.append(_render_svg(figure, 3))

            monthly_matrix, years = _monthly_returns(strategy_returns, dates)
            figure, axis = plt.subplots(figsize=(11, max(2.5, 0.48 * max(len(years), 1) + 1.8)))
            figures.append(figure)
            if len(years):
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "EOY"]
                finite = monthly_matrix[np.isfinite(monthly_matrix)]
                limit = max(float(np.max(np.abs(finite))), 0.01) if len(finite) else 0.01
                image = axis.imshow(monthly_matrix, aspect="auto", cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
                axis.set_xticks(np.arange(len(month_names)), labels=month_names)
                axis.set_yticks(np.arange(len(years)), labels=years)
                axis.set_title("Monthly Returns · Strategy Net", loc="left", weight="bold")
                for row in range(len(years)):
                    for column in range(13):
                        value = monthly_matrix[row, column]
                        if np.isfinite(value):
                            axis.text(column, row, f"{value:.1%}", ha="center", va="center", fontsize=8)
                figure.colorbar(image, ax=axis, label="Return")
            else:
                axis.text(0.5, 0.5, "No monthly return observations", ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
            figure.tight_layout()
            svg_charts.append(_render_svg(figure, 4))
        finally:
            for figure in figures:
                plt.close(figure)

    performance_rows = [
        ("Cumulative Return", _format_percent(strategy_metrics["cumulative_return"]), _format_percent(benchmark_metrics["cumulative_return"])),
        ("CAGR", _format_percent(strategy_metrics["cagr"]), _format_percent(benchmark_metrics["cagr"])),
        ("Annualized Volatility", _format_percent(strategy_metrics["annualized_volatility"]), _format_percent(benchmark_metrics["annualized_volatility"])),
        ("Sharpe Ratio", _format_number(strategy_metrics["sharpe_ratio"]), _format_number(benchmark_metrics["sharpe_ratio"])),
        ("Sortino Ratio", _format_number(strategy_metrics["sortino_ratio"]), _format_number(benchmark_metrics["sortino_ratio"])),
        ("Calmar Ratio", _format_number(strategy_metrics["calmar_ratio"]), _format_number(benchmark_metrics["calmar_ratio"])),
    ]
    risk_rows = [
        ("Skewness", _format_number(strategy_metrics["skewness"]), _format_number(benchmark_metrics["skewness"])),
        ("Excess Kurtosis", _format_number(strategy_metrics["excess_kurtosis"]), _format_number(benchmark_metrics["excess_kurtosis"])),
        ("Max Drawdown", _format_percent(strategy_metrics["max_drawdown"]), _format_percent(benchmark_metrics["max_drawdown"])),
        ("Max Drawdown Duration (days)", _format_number(strategy_metrics["max_drawdown_duration"], 0), _format_number(benchmark_metrics["max_drawdown_duration"], 0)),
        ("Historical Daily VaR 95%", _format_percent(strategy_metrics["daily_var_95"]), _format_percent(benchmark_metrics["daily_var_95"])),
        ("Daily CVaR 95%", _format_percent(strategy_metrics["daily_cvar_95"]), _format_percent(benchmark_metrics["daily_cvar_95"])),
    ]
    exposure_rows = [
        ("Beta", _format_number(strategy_metrics["beta"]), "n/a"),
        ("Annualized Alpha", _format_percent(strategy_metrics["annualized_alpha"]), "n/a"),
        ("Average Rebalance Turnover", _format_percent(strategy_metrics["average_rebalance_turnover"]), "n/a"),
        ("Annualized Transaction Cost Drag (bps/yr)", _format_number(strategy_metrics["annualized_transaction_cost_drag_bps"], 1), "n/a"),
        ("Annualized Margin Financing Drag (bps/yr)", _format_number(strategy_metrics["annualized_margin_financing_drag_bps"], 1), "n/a"),
        ("Average Gross Exposure", _format_percent(strategy_metrics["average_gross_exposure"]), "n/a"),
        ("Average Net Exposure", _format_percent(strategy_metrics["average_net_exposure"]), "n/a"),
    ]
    metrics_html = "".join((
        _metric_table("Performance & Efficiency", performance_rows, benchmark_name),
        _metric_table("Higher-Order Moments & Tail Risk", risk_rows, benchmark_name),
        _metric_table("Market Exposure & Trading Friction", exposure_rows, benchmark_name),
    ))

    title = f"Quantitative Risk & Performance Dashboard · {scenario_name}"
    chart_sections = "".join(
        f'<section class="chart chart-{index}">{svg}</section>'
        for index, svg in enumerate(svg_charts, start=1)
    )
    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root{{--ink:#1d2730;--muted:#65717b;--rule:#d8dee3;--paper:#f5f7f8;--blue:{STRATEGY_COLOR};--gold:{BENCHMARK_COLOR};}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:34px 28px 56px}} header{{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:26px}}
.eyebrow{{margin:0 0 7px;color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.11em;text-transform:uppercase}}
h1{{font-size:26px;line-height:1.2;margin:0 0 8px}} .subtitle{{color:var(--muted);margin:0}}
h2{{font-size:17px;margin:30px 0 12px}} h3{{font-size:13px;margin:0 0 10px}}
.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px;border-bottom:1px solid var(--rule);padding-bottom:22px}}
.metric-section{{min-width:0}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{padding:7px 6px;border-bottom:1px solid var(--rule);text-align:right;font-variant-numeric:tabular-nums}}
thead th{{color:var(--muted);font-weight:600}} th:first-child{{text-align:left}} tbody th{{font-weight:500}}
.legend{{display:flex;gap:18px;color:var(--muted);font-size:12px;margin:0 0 10px}} .legend span::before{{content:"";display:inline-block;width:10px;height:10px;margin-right:6px;border-radius:50%;background:var(--blue)}} .legend .benchmark::before{{background:var(--gold)}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .chart{{background:#fff;border:1px solid var(--rule);padding:10px;min-width:0;overflow:hidden}}
.chart-1,.chart-4{{grid-column:1 / -1}} .chart svg{{display:block;width:100%;height:auto}}
footer{{margin-top:24px;padding-top:12px;border-top:1px solid var(--rule);color:var(--muted);font-size:11px}}
@media(max-width:850px){{main{{padding:22px 14px}}.metrics{{grid-template-columns:1fr}}.charts{{grid-template-columns:1fr}}.chart-1,.chart-4{{grid-column:auto}}h1{{font-size:22px}}}}
</style>
</head>
<body><main>
<header><p class="eyebrow">Portfolio Research · Out-of-Sample</p><h1>{escape(title)}</h1><p class="subtitle">Net strategy performance, risk moments, exposure and execution friction versus benchmark.</p></header>
<div class="legend"><span>Strategy · net</span><span class="benchmark">{escape(benchmark_name)}</span></div>
<h2>Risk & Performance Summary</h2><div class="metrics">{metrics_html}</div>
<h2>Portfolio Diagnostics</h2><div class="charts">{chart_sections}</div>
<footer>Daily statistics use arithmetic returns; annualization assumes 252 trading days. Transaction-cost drag includes the configured linear and quadratic rebalance impact. Exposure statistics require optional rebalance weight history.</footer>
</main></body></html>"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_document, encoding="utf-8")
    return output
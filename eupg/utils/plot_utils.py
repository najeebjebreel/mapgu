"""
eupg.utils.plot_utils
=====================
Reusable plotting helpers for hyperparameter-sweep figures.

Typical usage from a notebook
------------------------------
    from eupg.utils.plot_utils import (
        apply_style, plot_sweep, plot_multi_sweep, plot_tradeoff,
    )

    apply_style()   # set CCS/NeurIPS-compatible rcParams once per session

    plot_sweep(
        x=K_VALS, util=UTIL, mia=MIA,
        orig_util=ORIG_UTIL, orig_mia=ORIG_MIA,
        x_label='$k$',
        util_label='Utility Acc',
        mia_label='MIA AUC',
        orig_util_label='Orig. utility',
        orig_mia_label='Orig. MIA AUC',
        legend_kw=dict(loc='lower right', ncol=2),
        ...
    )
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

# ── Shared figure dimensions ──────────────────────────────────────────────────
# All three plot functions default to the SAME (width, height) so that every
# saved PDF is identical in physical dimensions.  Override per-call with
# the figsize= kwarg when needed.
#
# For ACM CCS / IEEE two-column text (~3.33 in per column) a width of 2.3 in
# fits three figures side-by-side with \hfill separators.
DEFAULT_FIGSIZE      = (2.3, 2.1)   # width × height — no time sub-panel
DEFAULT_FIGSIZE_TIME = (2.3, 3.5)   # taller variant — with time sub-panel

# ── Shared line / marker sizes ────────────────────────────────────────────────
# Defined once so all three plot functions are visually consistent when
# figures are placed side-by-side in a LaTeX row.
LINE_WIDTH  = 1.4   # primary data lines
MARKER_SIZE = 4.5   # primary data markers

# ── Colour scheme (Wong colorblind-safe palette) ──────────────────────────────
C_UTIL     = '#0072B2'   # blue       — post-unlearning utility
C_MIA      = '#D55E00'   # vermillion — post-unlearning MIA AUC
C_REF_UTIL = '#444444'   # dark grey  — reference utility (dashed)
C_REF_MIA  = '#999999'   # mid grey   — reference MIA AUC (dotted)
C_TIME     = '#009E73'   # teal       — unlearning wall time

# Full Wong palette for multi-series plots
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']

# Line-style cycle for series — ensures B&W distinguishability
_LINE_STYLES = ['-', '--', '-.', ':']


def apply_style():
    """Apply NeurIPS/CCS-compatible matplotlib rcParams. Call once per session."""
    matplotlib.rcParams.update({
        # --- fonts ---
        'font.family':           'sans-serif',
        'font.sans-serif':       ['Helvetica', 'DejaVu Sans', 'Arial'],
        'font.size':             8,
        'axes.titlesize':        8,
        'axes.labelsize':        8,
        'xtick.labelsize':       7,
        'ytick.labelsize':       7,
        'legend.fontsize':       6,
        'legend.framealpha':     0.90,
        'legend.edgecolor':      '0.75',
        'legend.borderpad':      0.25,
        'legend.labelspacing':   0.18,
        'legend.handlelength':   1.4,
        'legend.handletextpad':  0.35,
        'legend.columnspacing':  0.6,
        # --- axes / ticks ---
        'axes.linewidth':        0.6,
        'xtick.major.width':     0.5,
        'ytick.major.width':     0.5,
        'xtick.major.size':      3.0,
        'ytick.major.size':      3.0,
        'xtick.direction':       'out',
        'ytick.direction':       'out',
        # --- lines / markers ---
        'lines.linewidth':       LINE_WIDTH,
        'lines.markersize':      MARKER_SIZE,
        'lines.markeredgewidth': 0.0,
        # --- grid (horizontal only — cleaner for sweep plots) ---
        'grid.color':            '0.88',
        'grid.linewidth':        0.4,
        'axes.grid':             True,
        'axes.grid.axis':        'y',
        # --- figure / saving ---
        'figure.dpi':            150,
        'savefig.dpi':           600,
        # embed fonts as Type 42 (TrueType) — required by IEEE/ACM submission
        'pdf.fonttype':          42,
        'ps.fonttype':           42,
    })


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_fig(fig, save_path, dpi):
    """Save figure at exact figsize dimensions.

    bbox_inches='tight' is intentionally omitted: it re-crops each PDF to its
    content bounding box, producing different physical sizes even when figsize
    is identical.  tight_layout(pad=0.30) — called by every plot function before
    _save_fig — already handles internal spacing, so omitting bbox_inches='tight'
    is safe and keeps all figures pixel-perfect for side-by-side LaTeX rows.
    """
    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f'Saved: {save_path}')


def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _apply_log_x(ax, x_ticks, x_ticklabels, tick_rotation=45):
    ax.set_xscale('log')
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        labels = x_ticklabels if x_ticklabels is not None else [str(int(v)) for v in x_ticks]
        ax.set_xticklabels(labels, rotation=tick_rotation, ha='right')


def _set_linear_ticks(ax, x_ticks, x_ticklabels):
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        if x_ticklabels is not None:
            ax.set_xticklabels(x_ticklabels)


def _resolve_figsize(figsize, has_time):
    """Return (width, height) from an explicit override or the shared defaults."""
    if figsize is not None:
        return figsize
    return DEFAULT_FIGSIZE_TIME if has_time else DEFAULT_FIGSIZE


# ── Public API ────────────────────────────────────────────────────────────────

def plot_sweep(
    x, util, mia,
    orig_util, orig_mia,
    *,
    x_label='',
    title='',
    util_label='Utility Acc',
    mia_label='MIA AUC',
    orig_util_label='Original utility Acc',
    orig_mia_label='Original MIA AUC',
    time_label='Unlearn time (s)',
    util_std=None,
    mia_std=None,
    time_s=None,
    time_std=None,
    x_ticks=None,
    x_ticklabels=None,
    log_x=False,
    ylim_q=None,
    ylim_t=None,
    legend_kw=None,
    time_legend_kw=None,
    figsize=None,
    save_path=None,
    dpi=600,
):
    """
    Single-series sweep plot (utility + MIA AUC, optional efficiency panel).

    Parameters
    ----------
    x, util, mia        : sweep values and post-unlearning metrics (%)
    orig_util, orig_mia : baseline metrics before unlearning
    util_label          : legend label for utility line
    mia_label           : legend label for MIA AUC line
    orig_util_label     : legend label for original-utility reference line
    orig_mia_label      : legend label for original-MIA reference line
    time_label          : legend label for unlearning-time line
    util_std, mia_std   : optional ±1σ arrays for shaded bands
    time_s              : optional unlearning wall time (s) — adds lower panel
    time_std            : optional ±1σ for time
    x_ticks             : explicit x-axis tick positions
    x_ticklabels        : explicit x-axis tick labels (defaults to str(x_ticks))
    log_x               : if True, use log scale on x-axis
    ylim_q / ylim_t     : (lo, hi) for quality / time panels
    legend_kw           : dict of kwargs forwarded to ax.legend() for the quality panel
    time_legend_kw      : dict of kwargs forwarded to ax.legend() for the time panel
    figsize             : (width, height) override; defaults to DEFAULT_FIGSIZE[_TIME]
    save_path           : path to save figure (None = do not save)
    dpi                 : resolution for saved figure
    """
    x    = np.asarray(x,    float)
    util = np.asarray(util, float)
    mia  = np.asarray(mia,  float)

    _q_legend_kw = dict(loc='best', ncol=1)
    if legend_kw:
        _q_legend_kw.update(legend_kw)

    _t_legend_kw = dict(loc='best')
    if time_legend_kw:
        _t_legend_kw.update(time_legend_kw)

    has_time  = time_s is not None
    _w, fig_h = _resolve_figsize(figsize, has_time)
    h_ratio   = [2.5, 1.5] if has_time else [1]

    fig, axs = plt.subplots(
        2 if has_time else 1, 1,
        figsize=(_w, fig_h), dpi=150,
        gridspec_kw=({'height_ratios': h_ratio, 'hspace': 0.10}
                     if has_time else {}),
    )
    ax_q = axs[0] if has_time else axs

    if log_x:
        _apply_log_x(ax_q, x_ticks, x_ticklabels)

    if util_std is not None:
        ax_q.fill_between(x, util - np.asarray(util_std, float),
                             util + np.asarray(util_std, float),
                          color=C_UTIL, alpha=0.12, zorder=2)
    if mia_std is not None:
        ax_q.fill_between(x, mia - np.asarray(mia_std, float),
                             mia + np.asarray(mia_std, float),
                          color=C_MIA, alpha=0.12, zorder=2)

    ax_q.plot(x, util, marker='o', color=C_UTIL, label=util_label,
              lw=LINE_WIDTH, markersize=MARKER_SIZE, zorder=4)
    ax_q.plot(x, mia,  marker='s', color=C_MIA,  label=mia_label,
              lw=LINE_WIDTH, markersize=MARKER_SIZE, zorder=4)
    ax_q.axhline(orig_util, color=C_REF_UTIL, ls='--', lw=0.9,
                 label=orig_util_label, zorder=3)
    ax_q.axhline(orig_mia,  color=C_REF_MIA,  ls=':',  lw=0.9,
                 label=orig_mia_label,  zorder=3)

    ax_q.set_ylabel('(%)')
    if title:
        ax_q.set_title(title)
    if ylim_q:
        ax_q.set_ylim(*ylim_q)

    if not log_x:
        _set_linear_ticks(ax_q, x_ticks, x_ticklabels)

    if not has_time:
        ax_q.set_xlabel(x_label)
    else:
        ax_q.tick_params(labelbottom=False)

    ax_q.legend(**_q_legend_kw)
    _style_ax(ax_q)

    if has_time:
        ax_t = axs[1]
        t = np.asarray(time_s, float)

        if log_x:
            _apply_log_x(ax_t, x_ticks, x_ticklabels)

        if time_std is not None:
            ts = np.asarray(time_std, float)
            ax_t.fill_between(x, t - ts, t + ts,
                              color=C_TIME, alpha=0.12, zorder=2)
        ax_t.plot(x, t, marker='D', color=C_TIME, label=time_label,
                  lw=LINE_WIDTH, markersize=MARKER_SIZE, zorder=4)
        ax_t.set_xlabel(x_label)
        ax_t.set_ylabel('Time (s)')
        if ylim_t:
            ax_t.set_ylim(*ylim_t)

        if not log_x:
            _set_linear_ticks(ax_t, x_ticks, x_ticklabels)

        ax_t.legend(**_t_legend_kw)
        _style_ax(ax_t)

    fig.tight_layout(pad=0.30)
    _save_fig(fig, save_path, dpi)
    plt.show()


def plot_multi_sweep(
    x, series,
    orig_util, orig_mia,
    *,
    x_label='',
    title='',
    orig_util_label='Original utility (Acc)',
    orig_mia_label='Original MIA AUC',
    util_suffix=' acc.',
    mia_suffix=' MIA',
    time_series=None,
    x_ticks=None,
    x_ticklabels=None,
    log_x=False,
    ylim_q=None,
    ylim_t=None,
    legend_kw=None,
    time_legend_kw=None,
    figsize=None,
    save_path=None,
    dpi=600,
):
    """
    Multi-series sweep (e.g. multiple methods or datasets).

    Parameters
    ----------
    x       : shared x-axis array
    series  : list of dicts, each with keys:
                  'util'       — utility array
                  'mia'        — MIA AUC array
                  'label'      — legend label prefix
                  'util_std'   — optional ±1σ (utility)
                  'mia_std'    — optional ±1σ (MIA AUC)
    orig_util, orig_mia  : baseline references (drawn as grey reference lines)
    orig_util_label      : legend label for original-utility reference
    orig_mia_label       : legend label for original-MIA reference
    util_suffix          : suffix appended to each series label for utility lines
    mia_suffix           : suffix appended to each series label for MIA lines
    time_series  : list of dicts ``{'time', 'label', 'time_std'}``
    legend_kw    : dict of kwargs forwarded to ax.legend() for the quality panel
    time_legend_kw : dict of kwargs forwarded to ax.legend() for the time panel
    log_x        : if True, use log scale on x-axis
    figsize      : (width, height) override; defaults to DEFAULT_FIGSIZE[_TIME]
    """
    x = np.asarray(x, float)

    _q_legend_kw = dict(loc='best', ncol=1)
    if legend_kw:
        _q_legend_kw.update(legend_kw)

    _t_legend_kw = dict(loc='best', ncol=1)
    if time_legend_kw:
        _t_legend_kw.update(time_legend_kw)

    has_time  = time_series is not None
    _w, fig_h = _resolve_figsize(figsize, has_time)
    h_ratio   = [2.5, 1.5] if has_time else [1]

    fig, axs = plt.subplots(
        2 if has_time else 1, 1,
        figsize=(_w, fig_h), dpi=150,
        gridspec_kw=({'height_ratios': h_ratio, 'hspace': 0.10}
                     if has_time else {}),
    )
    ax_q = axs[0] if has_time else axs

    if log_x:
        _apply_log_x(ax_q, x_ticks, x_ticklabels)

    markers = ['o', 's', 'D', 'v', 'p']

    for i, s in enumerate(series):
        c   = s.get('color', PALETTE[i % len(PALETTE)])
        lbl = s.get('label', f'series {i}')
        mk  = 'o' if 'EUPG' in lbl else 's'
        u   = np.asarray(s['util'], float)
        m   = np.asarray(s['mia'],  float)

        if 'util_std' in s:
            us = np.asarray(s['util_std'], float)
            ax_q.fill_between(x, u - us, u + us, color=c, alpha=0.10, zorder=2)
        if 'mia_std' in s:
            ms = np.asarray(s['mia_std'], float)
            ax_q.fill_between(x, m - ms, m + ms, color=c, alpha=0.10, zorder=2)

        ax_q.plot(
            x, u,
            marker=mk, color=c, ls='-',
            lw=LINE_WIDTH, markersize=MARKER_SIZE,
            label=f'{lbl}{util_suffix}',
            zorder=4,
        )
        ax_q.plot(
            x, m,
            marker=mk, color=c, ls='--',
            lw=LINE_WIDTH, markersize=MARKER_SIZE,
            label=f'{lbl}{mia_suffix}',
            zorder=4,
            markerfacecolor='white',
            markeredgewidth=1.2,
        )

    ax_q.axhline(
        orig_util, color=C_REF_UTIL, ls=':', lw=1.0,
        label=orig_util_label, zorder=3,
    )
    ax_q.axhline(
        orig_mia, color=C_REF_MIA, ls='-.', lw=1.0,
        label=orig_mia_label, zorder=3,
    )

    ax_q.set_ylabel('(%)')
    if title:
        ax_q.set_title(title)
    if ylim_q:
        ax_q.set_ylim(*ylim_q)

    if not log_x:
        _set_linear_ticks(ax_q, x_ticks, x_ticklabels)

    if not has_time:
        ax_q.set_xlabel(x_label)
    else:
        ax_q.tick_params(labelbottom=False)

    ax_q.legend(**_q_legend_kw)
    _style_ax(ax_q)

    if has_time:
        ax_t = axs[1]

        if log_x:
            _apply_log_x(ax_t, x_ticks, x_ticklabels)

        for i, ts_info in enumerate(time_series):
            c   = PALETTE[i % len(PALETTE)]
            mk  = markers[i % len(markers)]
            ls  = _LINE_STYLES[i % len(_LINE_STYLES)]
            t   = np.asarray(ts_info['time'], float)
            lbl = ts_info.get('label', f'series {i}')
            if 'time_std' in ts_info:
                tsd = np.asarray(ts_info['time_std'], float)
                ax_t.fill_between(x, t - tsd, t + tsd, color=c, alpha=0.10)
            ax_t.plot(x, t, marker=mk, color=c, ls=ls, label=lbl,
                      lw=LINE_WIDTH, markersize=MARKER_SIZE, zorder=4)

        ax_t.set_xlabel(x_label)
        ax_t.set_ylabel('Time (s)')
        if ylim_t:
            ax_t.set_ylim(*ylim_t)

        if not log_x:
            _set_linear_ticks(ax_t, x_ticks, x_ticklabels)

        ax_t.legend(**_t_legend_kw)
        _style_ax(ax_t)

    fig.tight_layout(pad=0.30)
    _save_fig(fig, save_path, dpi)
    plt.show()


def plot_tradeoff(
    series,
    *,
    title='',
    x_label='MIA AUC (%)',
    y_label='Utility Acc (%)',
    orig_util=None,
    orig_mia=None,
    orig_label='Original model',
    show_random_ref=True,
    random_ref_label='Random guess (MIA = 50%)',
    annotate=True,
    annot_fontsize=6.0,
    annot_offset=(4, 2),
    legend_kw=None,
    xlim=None,
    ylim=None,
    figsize=None,
    save_path=None,
    dpi=600,
):
    """
    Utility-privacy tradeoff scatter comparing two or more methods.

    Each point is one (k or epsilon) configuration; x = MIA AUC, y = Utility.
    Points closer to top-left (high utility, low MIA AUC) are preferred.

    Parameters
    ----------
    series : list of dicts, each with keys:
                 'util'         - list of utility values (%)
                 'mia'          - list of MIA AUC values (%)
                 'label'        - series name shown in the legend
                 'param_labels' - list of strings to annotate each point
                 'util_std'     - optional y error bars (1 sigma)
                 'mia_std'      - optional x error bars (1 sigma)
    orig_util, orig_mia : coordinates of the pre-unlearning reference point
    show_random_ref     : draw a vertical dotted line at MIA AUC = 50%
    annotate            : label each point with its param_labels value
    annot_offset        : (dx, dy) in points for annotation offset
    legend_kw           : dict of kwargs forwarded to ax.legend()
    xlim, ylim          : (lo, hi) axis limits; None = auto
    figsize             : (width, height) override; defaults to DEFAULT_FIGSIZE
    """
    _leg_kw = dict(loc='best')
    if legend_kw:
        _leg_kw.update(legend_kw)

    # plot_tradeoff never has a time sub-panel, so has_time=False always
    fig, ax = plt.subplots(figsize=_resolve_figsize(figsize, False), dpi=150)

    markers = ['o', '^', 's', 'D', 'v', 'p']

    for i, s in enumerate(series):
        c   = PALETTE[i % len(PALETTE)]
        mk  = markers[i % len(markers)]
        lbl = s.get('label', f'series {i}')
        u   = np.asarray(s['util'], float)
        m   = np.asarray(s['mia'],  float)

        xerr = np.asarray(s['mia_std'],  float) if 'mia_std'  in s else None
        yerr = np.asarray(s['util_std'], float) if 'util_std' in s else None

        ax.errorbar(
            m, u,
            xerr=xerr, yerr=yerr,
            fmt=mk, color=c, label=lbl,
            capsize=2, elinewidth=0.7, capthick=0.7,
            markersize=MARKER_SIZE, zorder=4,
        )
        order = np.argsort(m)
        # Ghost connector: intentionally subordinate — half the primary line weight
        ax.plot(m[order], u[order], color=c,
                lw=LINE_WIDTH * 0.5, alpha=0.40, zorder=3)

        if annotate and 'param_labels' in s:
            for xi, yi, lbl_pt in zip(m, u, s['param_labels']):
                ax.annotate(
                    str(lbl_pt), xy=(xi, yi),
                    xytext=annot_offset, textcoords='offset points',
                    fontsize=annot_fontsize, color=c,
                    va='bottom', ha='left',
                )

    if orig_util is not None and orig_mia is not None:
        ax.scatter(
            [orig_mia], [orig_util],
            marker='*', s=60, color=C_REF_UTIL,
            zorder=5, label=orig_label,
        )

    if show_random_ref:
        ax.axvline(50.0, color='0.65', ls=':', lw=0.8,
                   label=random_ref_label, zorder=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    ax.legend(**_leg_kw)
    _style_ax(ax)

    fig.tight_layout(pad=0.30)
    _save_fig(fig, save_path, dpi)
    plt.show()
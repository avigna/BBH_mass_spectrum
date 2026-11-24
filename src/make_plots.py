import pandas as pd
import numpy as np
import scipy as sc
from scipy import interpolate, integrate
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G, M_sun, R_sun
import astropy.units as u
from astropy.cosmology import z_at_value
AU_to_Rsun = (1 * u.AU).to(u.R_sun).value
import fnmatch
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import scipy.stats as st
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# load list of dataframes that contains pop-synth catalogs from the 4 channels
pop_list_file = Path(r"/.../pop_list.pkl")
with open(pop_list_file, "rb") as f:
    pop_list = pickle.load(f)
# list of names in the same order as in pop_list
gen_namelist = ["Triple",  "GC 1+2", "GC 1+1", "Binary"]

### this just removes the pisn gap systems from the 1st gen BBhs.
### while i think this channel is plausible, the way it's modelled in CMC is very simplistic way
remove_1st_mass_gap = True
if remove_1st_mass_gap:
    pop_listgc1p1 = pop_list[2]
    m_primary = pop_listgc1p1[['m1', 'm2']].max(axis=1)
    pop_list[2] = pop_listgc1p1[m_primary < 45].copy()

path2project = "/Users/andris/Dropbox/Documents/mass_ratio_gc_tr"
gen_namelist = ["Triple", "GC 1+2", "GC 1+1", "Binary"]
f_scale = [1., 1., 1., 1.]

# GWTC-4 -> should be updated to GWTC-5
GWevents = pd.read_csv(path2project + r"/lvk_cat.csv")

def get_chirp_mass(m1,m2):
    return (m1 * m2) ** (3./5) / ((m1 + m2) ** (1./5))


# ================= ============ =============== ==============
# =======  m vs q for all channels with contours + cmap =======
# ================= ============ =============== ==============

def make4x4plot(levels, m2plot_flag, cosmo_weights_on, marginals_on, save_mode):
    m2plot = None # just so that it's easier to realise if there's a typo

    fig = plt.figure(figsize=(11.69, 8.27))
    # divide the grid into a 2 x 2 space, i.e.
    # outer[0] | outer[1]
    # outer[2] | outer[3]

    outer = GridSpec(nrows = 2, ncols = 2, hspace = 0.28, wspace = 0.15,   # set it manually in interactive
                     top = 0.97, bottom = 0.07, left = 0.12, right = 0.95) #  mode and then print & save the values

    # --- binning the data for marginals ---
    nbins = 25
    bins_m = np.linspace(2.5, 100, nbins) # same for primary mass
    bins_q = np.linspace(0.0, 1.0, nbins)

    # --- loop over formation channels ---
    for i, pop_i in enumerate(pop_list):

        if marginals_on:
            #  divide outer[i] into a 2x2 grid
            #  inner[0, 0] = marginal | inner[0, 1] = empty
            #  inner[1, 0] = 2d dist  | inner[1, 1] = marginal
            inner = GridSpecFromSubplotSpec(
                nrows = 2, ncols = 2, subplot_spec = outer[i],
                width_ratios=[3, 1],  # [X, 1]: left column is X times wider than right column
                height_ratios=[1, 3],  # [1, X]: lower row is X times taller than upper row
                wspace = 0.12, hspace = 0.12
            )
            ax_top  = fig.add_subplot(inner[0, 0]) # q - marginal
            ax_main = fig.add_subplot(inner[1, 0], sharex=ax_top) #  q (x-axis) vs m (y-axis) dist
            ax_left = fig.add_subplot(inner[1, 1], sharey=ax_main) # m - marginal
        else:
            ax_main = fig.add_subplot(outer[i])

        # --- prep pop synthdata ---
        m_primary = pop_i[['m1', 'm2']].max(axis=1).values
        m_secondary = pop_i[['m1', 'm2']].min(axis=1).values
        q = m_secondary / m_primary
        if m2plot_flag == "mchirp":
            m2plot = get_chirp_mass(m_primary, m_secondary)
        elif m2plot_flag == "m1":
            m2plot = m_primary

        if cosmo_weights_on:
            weights = pop_i['cosmic weight (z = 0)'].values
        else:
            weights = np.ones_like(pop_i['cosmic weight (z = 0)'].values)

        # --- kde fitting ---
        qmin, qmax = 0.0, 1.0
        m1_min, m1_max = 2.5, 60.9
        m_grid = np.linspace(m1_min, m1_max, 100)
        q_grid = np.linspace(qmin, qmax, 100)
        qq, mm = np.meshgrid(q_grid, m_grid, indexing = "ij")
        # in st. gaussian_kdefor mult-d kde data has to be formatted such that (n_dimensions, n_datapoints)
        positions = np.vstack([qq.reshape(-1), mm.reshape(-1)]) # positions for which we evaluate the kde
        values = np.vstack([q, m2plot]) # we're fitting the kde to these datapoints
        kernel = st.gaussian_kde(values, weights=weights)
        qm_pdf = np.reshape(kernel(positions).T, qq.shape) # evaluate kerner and reshape it to 100 x 100

        # --- finding the contours ---
        norm_qm_pdf = qm_pdf / qm_pdf.sum()
        pdf_sorted = np.sort(norm_qm_pdf.reshape(-1))[::-1] # flipping backwards, going from high density to low density
        cdf = np.cumsum(pdf_sorted)
        # search the closest value where cdf == level_value and we check what is the value of pdf_sorted there
        levels_cont_list = []
        for level in levels:
            levels_cont_list.append(pdf_sorted[np.searchsorted(cdf, level)])
        levels_cont_list = np.sort(levels_cont_list) # we need to sort this, otherwise contourf will complain

        # --- plot cmap & contours ---
        ax_main.contourf(qq, mm, qm_pdf, cmap='Blues')
        ax_main.contour(qq, mm, norm_qm_pdf, levels=levels_cont_list, colors="k")

        if marginals_on:
            # --- marginals for q ---
            counts, _ = np.histogram(q, bins=bins_q,  weights=weights)
            bin_width = np.diff(bins_q)[0]
            norm_scale = np.sum(counts * bin_width) ** -1 * weights # norm to merger rate desnity or to 1, depending on cosmo_weights_on
            ax_top.hist(q, bins=bins_q, histtype="step", linewidth=2.5, color="black", weights=norm_scale)
            # --- marignals for mass ---
            counts, _ = np.histogram(m2plot, bins=bins_m, weights=weights)
            bin_width = np.diff(bins_m)[0]
            norm_scale =  np.sum(counts * bin_width) ** -1 * weights
            ax_left.hist(m2plot, bins=bins_m, histtype="step", linewidth=2.5, color="black", weights=norm_scale, orientation="horizontal" )

        # --- plotting gwtc sources ---
        m_pr_obs = GWevents["mass_1_source"]
        #m_sc_obs = GWevents["mass_2_source"]
        q_obs = GWevents["q"]

        if m2plot_flag == "mchirp":
            m2plot_obs = GWevents["chirp_mass_source"]
        elif m2plot_flag == "m1":
            m2plot_obs = m_pr_obs

        ax_main.plot(q_obs, m2plot_obs, linestyle="None", marker="o",  color="gold",
            markeredgecolor="k", markeredgewidth=0.5, alpha=0.6,markersize=3
        )
        #  --- error bars ---
        if m2plot_flag == "mchirp":
            lower_m = np.abs(GWevents["chirp_mass_source_lower"].values)
            upper_m = GWevents["chirp_mass_source_upper"].values
            merr = np.vstack((lower_m, upper_m))
        elif m2plot_flag == "m1":
            lower_m = np.abs(GWevents["mass_1_source_lower"].values)
            upper_m = GWevents["mass_1_source_upper"].values
            merr = np.vstack((lower_m, upper_m))
        lower_q = np.abs(GWevents["q_lower"].values)
        upper_q = GWevents["q_upper"].values
        qerr = np.vstack((lower_q, upper_q))
        ax_main.errorbar(q_obs, m_pr_obs, xerr = qerr, yerr = merr, fmt = "none", ecolor = "black",
                         alpha = 0.1, elinewidth = 0.5, capsize = 0.2)

        #  ---style stuff ---
        ax_main.text(0.05, 0.95, gen_namelist[i], transform=ax_main.transAxes,
            fontsize=14, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
        ax_main.set_ylim(0, 60)
        ax_main.tick_params(axis="both", labelsize=14)
        if marginals_on:
            # --- q marginal (top) ---
            ax_top.tick_params(axis="x", labelbottom=False)
            ax_top.tick_params(axis="y", labelleft=False, left=False)
            ax_top.set_yscale("log")
            ax_top.set_ylim([10 ** -1, 10 ** 2])

            ax_top.set_ylabel(r"$\log P(q)\;[\mathrm{a.u.}]$", fontsize=8,
                              rotation=90, labelpad=8)
            ax_top.yaxis.set_label_position("right")

            # --- mass marginal (right) ---
            ax_left.tick_params(axis="x", labelbottom=False, bottom=False)
            ax_left.tick_params(axis="y", labelleft=False)
            ax_left.set_xscale("log")
            ax_left.set_xlim([10 ** -2, 10 ** 1])

            if m2plot_flag == "mchirp":
                mass_label = r"$\log P(M_{\rm chirp})\;[\mathrm{a.u.}]$"
            elif m2plot_flag == "m1":
                mass_label = r"$\log P(m_1)\;[\mathrm{a.u.}]$"

            ax_left.set_xlabel(mass_label, fontsize=8, labelpad=8)

        fig.supxlabel(r"$q = m_2/m_1$", fontsize=14)
        if m2plot_flag == "mchirp":
            fig.supylabel(r"$M_{\rm chirp}\,[M_{\odot}]$", fontsize=14)
        elif m2plot_flag == "m1":
            fig.supylabel(r"$M_{\rm BH,1}\,[M_{\odot}]$", fontsize=14)

    if save_mode:
        file_name = "weighted_" if cosmo_weights_on else "unweighted_"
        file_name += "q_vs_"
        file_name += "m1_" if m2plot_flag == "m1" else "mchirp_"
        file_name += "with_marginals" if marginals_on else ""
        file_name += "with_contours_" + "_".join([str(level).replace(".", "d") for level in levels])
        file_name += ".pdf"
        fig.savefig(file_name, format="pdf", dpi=300, bbox_inches="tight")


# --- add switches ---
m2plot_flag = "m1" # 'm1': primary mass, 'mchirp': chirp mass
marginals_on = False # True, False
cosmo_weights_on = True # True, False
save_mode = True

# amke 4 different versions
make4x4plot(levels = [0.925],
            m2plot_flag = "m1",
			cosmo_weights_on = False,
			marginals_on = marginals_on,
			save_mode = True)

make4x4plot(levels = [0.925],
            m2plot_flag = "m1",
			cosmo_weights_on = True,
			marginals_on = marginals_on,
			save_mode = True)

make4x4plot(levels = [0.925],
            m2plot_flag = "mchirp",
			cosmo_weights_on = False,
			marginals_on = marginals_on,
			save_mode = True)

make4x4plot(levels = [0.925],
            m2plot_flag = "mchirp",
			cosmo_weights_on = True,
			marginals_on = marginals_on,
			save_mode = True)

make4x4plot(levels = [0.9, 0.925, 0.95],
            m2plot_flag = "m1",
			cosmo_weights_on = False,
			marginals_on = marginals_on,
			save_mode = True)


# --- fig with mass distributions with different m1 intervals
def plot_q_dist_per_mass(m1_bins, channel_colors, fractions_on, save_mode):
    # predefine stuff for normalisation and printing fractions
    f_scale = [np.round(pop['cosmic weight (z = 0)'].sum(), 3) for pop in pop_list]
    rtot = np.array(f_scale).sum()
    f_f = np.array([np.round(f_i/rtot, 3) for f_i in f_scale])
    f_per_massbin_all = []
    f_per_massbin_q_all = []

    fig, axes = plt.subplots(3, 2, figsize=(6.51 * 2, 7.33), sharex=False)

    q_grid = np.linspace(0, 1, 500)
    q_zoom = q_grid <= 0.7
    for row, (m1_min, m1_max) in enumerate(m1_bins):
        q_kdes, labels, colors = [], [], []
        f_per_massbin = []
        f_per_massbin_q = []

        for i, pop_i in enumerate(pop_list):
            m_primary = pop_i[['m1', 'm2']].max(axis=1).values
            m_secondary = pop_i[['m1', 'm2']].min(axis=1).values
            q = m_secondary / m_primary
            weights = pop_i['cosmic weight (z = 0)'].values

            if m1_min == m1_low or m1_max != m1_high: # masks for filtering out the masses that belong to the given mass interval
                mask = (m_primary > m1_min) & (m_primary <= m1_max)
            else:
                mask = (m_primary > m1_min) & (m_primary < m1_max)
            mask_q = (m_secondary/m_primary < 0.7) # mask for "low_q" systems -> this is only needed for printing fractions in the right panels
            q_selected, weights_selected = q[mask], weights[mask]
            # collecting out the relative rates for each channel for all and for q< 0.
            f_per_massbin.append(np.round(weights[mask].sum(), 3))
            f_per_massbin_q.append(np.round(weights[mask & mask_q].sum(), 3))
            if len(q_selected) < 2 or np.std(q_selected) == 0:
                # just skip if a channel is not present in a mass bin
                continue
            bandwidth = 0.04
            #if (row == 0) | (row == 1):
            #    bandwidth = 0.04
            #else:
            #    bandwidth = 0.04 # 0.02

            # q_mean and q_var needed for the bandwidth calculation
            q_mean = np.average(q_selected, weights=weights_selected)
            q_var = np.sum(weights_selected * (q_selected - q_mean)**2) / (np.sum(weights_selected) - np.sum(weights_selected**2) / np.sum(weights_selected))
            # fitting the kde
            kde = st.gaussian_kde(q_selected, weights=weights_selected, bw_method=bandwidth / np.sqrt(q_var))
            kde_values = 0.01 * f_scale[i] * kde(q_grid)
            q_kdes.append(kde_values)
            labels.append(gen_namelist[i])
            colors.append(channel_colors[i])

        ### collecting stuff to calculate the fractions
        f_per_massbin_all.append(f_per_massbin)
        f_per_massbin_q_all.append(f_per_massbin_q)
        rtot_per_massbin = np.array(f_per_massbin).sum()
        f_f_per_massbin = np.array([np.round(f_i / rtot_per_massbin, 3) for f_i in f_per_massbin])
        # same but for q<0.7
        rtot_per_massbin_q = np.array(f_per_massbin_q).sum()
        f_f_per_massbin_q = np.array([np.round(f_i / rtot_per_massbin_q, 3) for f_i in f_per_massbin_q])

        # --- stacking the fitted kdes
        for col, ax in enumerate(axes[row]):
            cumulative = np.zeros_like(q_grid)
            for kde_values, label, color in zip(q_kdes, labels, colors):
                ax.fill_between(q_grid, cumulative, cumulative + kde_values,color=color, alpha=0.85, label=label)
                cumulative += kde_values

            # Zoom in on q < 0.7 in the right column
            if col == 1:
                ax.set_xlim(0, 0.7)
                ax.set_ylim(0, 1.05 * np.max(cumulative[q_zoom]))
            else:
                ax.set_xlim(0, 1)

            # --- just printing the i) mass intevals and the ii) channel  fractions
            m1_symbol = r"\leq" if m1_max != m1_high else "<"
            m1_label = rf"${m1_min} < m_1 {m1_symbol} {m1_max}\,M_\odot$"
            ax.text(0.05, 0.90, m1_label, transform=ax.transAxes, fontsize=14, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            if fractions_on:
                # fractions
                frac_text = "\n".join( f"{name} = {frac:.2f}" for name, frac in zip(gen_namelist, f_f_per_massbin))
                axes[row, 0].text(0.02, 0.2, frac_text, transform=axes[row, 0].transAxes,
                                  ha="left", va="bottom", fontsize=10,
                                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                # fractions on the zoom-in panels (left)
                frac_text = "\n".join(f"{name} = {frac:.2f}" for name, frac in zip(gen_namelist, f_f_per_massbin_q))
                axes[row, 1].text(0.02, 0.2, frac_text, transform=axes[row, 1].transAxes, ha="left", va="bottom",
                                  fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_ylabel(r"$p(q)$ [a.u.]", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)


    for ax in axes[-1]:
        ax.set_xlabel(r"$q = m_2/m_1$", fontsize=16)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncols=4, fontsize=10, frameon=False)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.07, right=0.98, hspace=0.35, wspace=0.20)

    if save_mode:
        fig.savefig("kde_weighted_q_m1_bins_A4_stacked_zoom.pdf", format="pdf", dpi=300, bbox_inches="tight")

m1_low = 4
m1_mid1 = 15
m1_mid2 = 46
m1_high = 60
m1_bins = [(m1_low, m1_mid1), (m1_mid1, m1_mid2), (m1_mid2, m1_high)]
channel_colors = ["#386CB0", "#1B9E77", "#E6AB02", "#7570B3"]

plot_q_dist_per_mass(m1_bins = m1_bins, channel_colors = channel_colors, fractions_on = True, save_mode = True)
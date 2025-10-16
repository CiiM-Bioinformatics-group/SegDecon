# -*- coding: utf-8 -*-
# A1: Head-to-head segmentation performance (tile-level counts)

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product, combinations
from scipy import stats as st


base_tpl = './meta/result_{method}/{slide}/{slide}_counts_compare_{method}.csv'
methods = ['segdecon', 'spotiphy', 'stardist']
slides  = ['img0', 'img1']   # img0=internal, img1=external
output_dir = '/vol/projects/yxi24/SegDecon/REVERSE/analysis_A1'
os.makedirs(output_dir, exist_ok=True)

# “Spotiphy + SegDecon HSV”  tweak）
path_spotiphy_hsv = './meta/result_spotiphy_with_segdecon/img0_counts_compare_spotiphy.csv'


dfs = []
for method, slide in product(methods, slides):
    p = base_tpl.format(method=method, slide=slide)
    df = pd.read_csv(p)
    
    need = {'tile_id','filename','count_gt'}
   
    assert need.issubset(set(df.columns)), f'缺列: {need - set(df.columns)} in {p}'
    
    if 'pred_count' in df.columns:
        df['err'] = df.get('err', df['pred_count'] - df['count_gt'])
    else:
        raise ValueError(f'未发现 pred_count in {p}')
    df['pred'] = df['pred_count'].astype(float)
    df['gt']   = df['count_gt'].astype(float)
    df['abs_err'] = df['err'].abs()
    df['method'] = method
    df['slide']  = slide
   
    df['key'] = df['slide'].astype(str) + '|' + df['filename'].astype(str)
    dfs.append(df[['key','slide','method','filename','tile_id','gt','pred','err','abs_err']])

all_df = pd.concat(dfs, ignore_index=True)


eps = 1.0  
summary_rows = []


density_bins = {}
for slide in slides:
    
    gt_unique = (all_df[all_df['slide']==slide]
                 .drop_duplicates(subset=['key'])[['key','gt']])
    qs = gt_unique['gt'].quantile([0.25,0.5,0.75]).to_list()
    bins = [-np.inf, qs[0], qs[1], qs[2], np.inf]
    labels = ['Q1','Q2','Q3','Q4']
    density_bins[slide] = (bins, labels)


for slide in slides:
    for method in methods:
        g = all_df[(all_df['slide']==slide) & (all_df['method']==method)]
        if g.empty: continue
        me   = g['err'].mean()
        mpe  = 100.0 * np.mean(g['err'] / np.maximum(g['gt'], eps))
        mae  = g['abs_err'].mean()
        rmse = math.sqrt(np.mean(g['err']**2))
        mape = 100.0 * np.mean(np.abs(g['err']) / np.maximum(g['gt'], eps))
        r = np.corrcoef(g['pred'], g['gt'])[0,1] if g['pred'].std()>0 and g['gt'].std()>0 else np.nan
        r2 = r**2 if pd.notnull(r) else np.nan
        diff = g['pred'] - g['gt']
        mean_diff = diff.mean()
        sd_diff   = diff.std(ddof=1)
        loa_low   = mean_diff - 1.96*sd_diff
        loa_high  = mean_diff + 1.96*sd_diff
        summary_rows.append(
            dict(level='by_slide', slide=slide, method=method,
                 n=g.shape[0], ME=me, MPE=mpe, MAE=mae, RMSE=rmse, MAPE=mape,
                 R2=r2, BA_mean_diff=mean_diff, BA_loa_low=loa_low, BA_loa_high=loa_high)
        )


for method in methods:
    g = all_df[all_df['method']==method]
    me   = g['err'].mean()
    mpe  = 100.0 * np.mean(g['err'] / np.maximum(g['gt'], eps))
    mae  = g['abs_err'].mean()
    rmse = math.sqrt(np.mean(g['err']**2))
    mape = 100.0 * np.mean(np.abs(g['err']) / np.maximum(g['gt'], eps))
    r = np.corrcoef(g['pred'], g['gt'])[0,1] if g['pred'].std()>0 and g['gt'].std()>0 else np.nan
    r2 = r**2 if pd.notnull(r) else np.nan
    diff = g['pred'] - g['gt']
    mean_diff = diff.mean()
    sd_diff   = diff.std(ddof=1)
    loa_low   = mean_diff - 1.96*sd_diff
    loa_high  = mean_diff + 1.96*sd_diff
    summary_rows.append(
        dict(level='combined', slide='img0+img1', method=method,
             n=g.shape[0], ME=me, MPE=mpe, MAE=mae, RMSE=rmse, MAPE=mape,
             R2=r2, BA_mean_diff=mean_diff, BA_loa_low=loa_low, BA_loa_high=loa_high)
    )

summary_df = pd.DataFrame(summary_rows)

#bland altman summary
method_order  = ['segdecon', 'stardist']
method_colors = {'segdecon':'#274753', 'stardist':'#e66d50'}


slide_order   = ['img0', 'img1']  # img0: internal, img1: external
slide_labels  = {'img0':'Internal', 'img1':'External'}

def _agg(df):
    diff = df['pred'] - df['gt']
    md   = diff.mean()
    sd   = diff.std(ddof=1)
    loa_low, loa_high = md - 1.96*sd, md + 1.96*sd
    return pd.Series({'BA_mean_diff':md,'BA_loa_low':loa_low,'BA_loa_high':loa_high})

agg = (all_df[all_df['slide'].isin(slide_order) & all_df['method'].isin(method_order)]
       .groupby(['slide','method'], as_index=False)
       .apply(lambda g: _agg(g.droplevel(0) if isinstance(g.index, pd.MultiIndex) else g))
       .reset_index(drop=True))

#MAE
method_order  = ['segdecon', 'stardist']
method_colors = {'segdecon':'#274753', 'stardist':'#e66d50'}
slide_order   = ['img0', 'img1']
slide_labels  = {'img0':'Internal', 'img1':'External'}

df = all_df.copy()
df = df[df['slide'].isin(slide_order) & df['method'].isin(method_order)].copy()
if 'abs_err' not in df.columns and {'pred','gt'}.issubset(df.columns):
    df['abs_err'] = (df['pred'] - df['gt']).abs()
#Density-stratified tile-wise absolute count errors
method_order  = ['segdecon', 'spotiphy', 'stardist']
method_colors = {'segdecon':'#274753','spotiphy':'#e7c66b','stardist':'#e66d50'}
slide_labels  = {'img0':'data1 (internal)', 'img1':'data2 (external)'}
alpha_box     = 0.90  
jitter_sigma  = 0.06  
eps           = 1.0  

slides_to_plot = [s for s in ['img0','img1'] if s in all_df['slide'].unique()]

for slide in slides_to_plot:
   
    gt_unique = (all_df[all_df['slide']==slide]
                 .drop_duplicates(subset=['key'])[['key','gt']]).copy()
    q1,q2,q3 = gt_unique['gt'].quantile([0.25, 0.50, 0.75]).tolist()
    bins   = [-np.inf, q1, q2, q3, np.inf]
    labels = ['Q1 (sparse)','Q2','Q3','Q4 (dense)']
    gt_unique['density_bin'] = pd.cut(gt_unique['gt'], bins=bins, labels=labels, include_lowest=True)

   
    tmp = (all_df[all_df['slide']==slide]
           .merge(gt_unique[['key','density_bin']], on='key', how='left'))

    
    data = []            # （|err|）
    positions = []       # 
    box_methods = []     #
    bin_ns = []          # 

    n_methods = len(method_order)
    group_gap = 1.6      # 
    box_width = 0.6      # 

   
    bin_counts = (gt_unique.groupby('density_bin', observed=True)
                  .size().reindex(labels, fill_value=0).tolist())

    for b_idx, dbin in enumerate(labels):
       
        base = b_idx * (n_methods + group_gap)
        bin_ns.append(bin_counts[b_idx])

        for m_idx, m in enumerate(method_order):
            s = tmp[(tmp['density_bin']==dbin) & (tmp['method']==m)]['abs_err'].dropna().values
            data.append(s)
            positions.append(base + m_idx)     
            box_methods.append(m)
#spotiphy+segdecon preprocessing
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats as st


slide = 'img0'
slide_label = 'data1 (internal)'

# HSV +Spotiphy
path_spotiphy_hsv = './meta/result_spotiphy_with_segdecon/img0_counts_compare_spotiphy.csv'


method_order   = ['spotiphy', 'spotiphy_hsv']  
method_colors  = {'spotiphy':'#e7c66b', 'spotiphy_hsv':'#165188'}
alpha_box      = 0.90
jitter_sigma   = 0.06


assert os.path.exists(path_spotiphy_hsv), "HSV CSV not found."

df_hsv = pd.read_csv(path_spotiphy_hsv)
df_hsv['pred'] = df_hsv['pred_count'].astype(float)
df_hsv['gt']   = df_hsv['count_gt'].astype(float)
df_hsv['err']  = df_hsv.get('err', df_hsv['pred'] - df_hsv['gt'])
df_hsv['abs_err'] = df_hsv['err'].abs()
df_hsv = df_hsv[['filename','gt','abs_err']].rename(columns={'gt':'gt_HSV','abs_err':'abs_err_HSV'})

base_sp = (all_df[(all_df['slide']==slide) & (all_df['method']=='spotiphy')]
           [['filename','gt','abs_err']].rename(columns={'gt':'gt_base','abs_err':'abs_err_base'}))

mrg = pd.merge(base_sp, df_hsv, on='filename', how='inner')


gt_unique = (all_df[all_df['slide']==slide]
             .drop_duplicates(subset=['key'])[['key','gt']]).copy()
q1,q2,q3 = gt_unique['gt'].quantile([0.25, 0.50, 0.75]).tolist()
bins   = [-np.inf, q1, q2, q3, np.inf]
labels = ['Q1 (sparse)','Q2','Q3','Q4 (dense)']


fname2key = (all_df[all_df['slide']==slide][['filename','key']]
             .drop_duplicates(subset=['filename']))
mrg = mrg.merge(fname2key, on='filename', how='left')
mrg = mrg.merge(gt_unique.rename(columns={'gt':'gt_for_bin'}), on='key', how='left')
mrg['density_bin'] = pd.cut(mrg['gt_for_bin'], bins=bins, labels=labels, include_lowest=True)


data      = []      
positions = []      
who       = []     
bin_ns    = []     

n_methods  = len(method_order)  # 2
group_gap  = 1.8
box_width  = 0.6


bin_counts = (mrg.groupby('density_bin', observed=True)
              .size().reindex(labels, fill_value=0).tolist())

for b_idx, dbin in enumerate(labels):
    base = b_idx*(n_methods + group_gap)
    bin_ns.append(bin_counts[b_idx])

    sub = mrg[mrg['density_bin']==dbin]
    # baseline
    data.append(sub['abs_err_base'].dropna().values)
    positions.append(base + 0)
    who.append('spotiphy')
    # hsv
    data.append(sub['abs_err_HSV'].dropna().values)
    positions.append(base + 1)
    who.append('spotiphy_hsv')


fig, ax = plt.subplots(figsize=(10, 5))

bp = ax.boxplot(
    data, positions=positions, widths=box_width, showfliers=False, patch_artist=True,
    boxprops=dict(edgecolor='#444444', linewidth=1.2),
    medianprops=dict(color='#777777', linewidth=1.6),
    whiskerprops=dict(color='#444444', linewidth=1.0),
    capprops=dict(color='#444444', linewidth=1.0)
)


for patch, m in zip(bp['boxes'], who):
    patch.set_facecolor(method_colors[m])
    patch.set_alpha(alpha_box)
    patch.set_zorder(2)


for pos, vals, m in zip(positions, data, who):
    if len(vals)==0: continue
    x = np.random.normal(loc=pos, scale=jitter_sigma, size=len(vals))
    ax.scatter(x, vals, s=12, alpha=0.6, color=method_colors[m],
               edgecolors='#000000', linewidths=0.3, zorder=5)


group_centers = [b*(n_methods + group_gap) + 0.5 for b in range(len(labels))]
xticklabels = [f"{lab}\n[n={n}]" for lab, n in zip(labels, bin_ns)]
ax.set_xticks(group_centers)
ax.set_xticklabels(xticklabels)

ax.set_ylabel('|err| per tile')
ax.set_title(f"Density-stratified |err|  •  Internal", fontsize=14)
ax.grid(axis='y', linestyle=':', color='#dddddd', linewidth=0.8, alpha=0.8)


handles = [Patch(facecolor=method_colors[m], edgecolor='#444444', alpha=alpha_box, label=m.replace('_',' + '))
           for m in method_order]
ax.legend(handles=handles, frameon=False, ncol=2, loc='upper right')


def star_from_p(p):
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 5e-2: return '*'
    return 'ns'

def add_sig_bracket(ax, x1, x2, y, text, dh=0.02, lw=1.0):
    """在 x1 与 x2 上方 y 位置画括号并标注 text。dh 为额外高度比例（相对y轴范围）。"""
    ylim = ax.get_ylim()
    h = dh*(ylim[1]-ylim[0])
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='#333333', linewidth=lw)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=10, color='#333333')


y_max_per_bin = []
for b_idx, dbin in enumerate(labels):
    vals_before = data[2*b_idx]
    vals_after  = data[2*b_idx+1]
    y_max_per_bin.append(np.nanmax([np.max(vals_before) if len(vals_before)>0 else 0,
                                    np.max(vals_after)  if len(vals_after)>0  else 0]))


for b_idx, dbin in enumerate(labels):
    sub = mrg[mrg['density_bin']==dbin][['filename','abs_err_base','abs_err_HSV']].dropna()
    if sub.empty:
        continue
   
    s = sub.set_index('filename')
    x = s['abs_err_base'].values
    y = s['abs_err_HSV'].values
    try:
        w = st.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided', method='auto')
        p = w.pvalue
    except ValueError:
        p = np.nan
    stars = star_from_p(p) if not np.isnan(p) else 'NA'

  
    x1 = b_idx*(n_methods + group_gap) + 0
    x2 = b_idx*(n_methods + group_gap) + 1
    y0 = y_max_per_bin[b_idx]
  
    y_br = y0 + 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0])
    add_sig_bracket(ax, x1, x2, y_br, stars, dh=0.01, lw=1.0)


ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax*1.10 if ymax>0 else ymax+1.0)

plt.tight_layout()



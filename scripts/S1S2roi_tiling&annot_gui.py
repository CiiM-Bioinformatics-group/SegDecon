#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle

import pandas as pd
import matplotlib as mpl
import scanpy as sc
import torch


import numpy as np
import matplotlib.pyplot as plt
import squidpy as sq
import warnings
warnings.filterwarnings('ignore')


import anndata
import geopandas as gpd
from tifffile import imread, imwrite
from shapely.geometry import Polygon, Point
from scipy import sparse
from matplotlib.colors import ListedColormap
import cv2
from skimage.io import imread
from skimage.measure import find_contours
import seaborn as sns
import json
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm



# In[2]:


import gc


# In[3]:


gc.collect()


# In[4]:


img = imread('./Visium_HD_Mouse_Brain_tissue_image.tif', plugin='tifffile')
img1 = imread('./CytAssist_Fresh_Frozen_Sagittal_Mouse_Brain_tissue_image.tif', plugin='tifffile')


# In[5]:


crop_img = img[800:3800, 15000:18000]
crop_img1 = img1[17500:20500, 7000:10000]


# In[12]:


plt.imshow(crop_img)


# In[11]:


plt.imshow(crop_img1)




import numpy as np, os, pandas as pd
from PIL import Image

def np_to_pil(arr: np.ndarray) -> Image.Image:
    """把 numpy 数组（灰度/uint8/uint16/RGB）统一成 RGB 的 PIL.Image"""
    if arr.ndim == 2:  # 灰度 -> 3通道
        arr = np.stack([arr]*3, axis=-1)

    
    if arr.dtype == np.uint8:
        rgb = arr
    elif arr.dtype == np.uint16:
       
        rgb = (arr / 257).astype(np.uint8)
    else:
       
        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max == a_min:
            rgb = np.zeros_like(arr, dtype=np.uint8)
        else:
            rgb = ((arr - a_min) / (a_max - a_min) * 255).astype(np.uint8)

    return Image.fromarray(rgb).convert('RGB')



img = np_to_pil(crop_img1)           
# img = np_to_pil(crop_img)         
W, H = img.size
tile_size = 500
cols = W // tile_size
rows = H // tile_size


newW, newH = cols * tile_size, rows * tile_size
x0, y0 = (W - newW)//2, (H - newH)//2
img_crop = img.crop((x0, y0, x0+newW, y0+newH))

os.makedirs('tiles', exist_ok=True)
os.makedirs('meta', exist_ok=True)

records, tid = [], 0
for r in range(rows):
    for c in range(cols):
        left, top = c*tile_size, r*tile_size
        box = (left, top, left+tile_size, top+tile_size)
        tile = img_crop.crop(box)
        name = f"tile_r{r:02d}_c{c:02d}_id{tid:02d}.png"
        tile.save(os.path.join('tiles', name))
        records.append({
            'tile_id': tid, 'r': r, 'c': c, 'filename': name,
            'x': int(left + x0), 'y': int(top + y0),
            'w': tile_size, 'h': tile_size
        })
        tid += 1


pd.DataFrame(records).to_csv('meta/tile_index.csv', index=False)
print(f"Saved {tid} tiles -> tiles/, index -> meta/tile_index.csv; grid = {rows}x{cols} (tile={tile_size}x{tile_size})")


# In[5]:


get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib
print(matplotlib.get_backend())  # 期待: module://ipympl.backend_nbagg

import matplotlib.pyplot as plt
plt.figure(); plt.plot([0,1,2],[0,1,0]); plt.title("ipympl ok");


# In[6]:


get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib; print(matplotlib.get_backend())  # 应显示 module://ipympl.backend_nbagg


# In[7]:


import sys, matplotlib
print('python =', sys.executable)
print('backend =', matplotlib.get_backend())

try:
    import jupyterlab, ipywidgets, ipympl
    print('jupyterlab =', jupyterlab.__version__)
    print('ipywidgets =', ipywidgets.__version__)
    print('ipympl =', ipympl.__version__)
except Exception as e:
    print('import error:', e)


get_ipython().run_line_magic('matplotlib', 'widget')

import os, numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display


CROP_X0 = 15000   
CROP_Y0 = 800    

# ---------- path ----------
TILES_DIR = 'tiles/img0'
INDEX_CSV = 'meta/img0_tile_index.csv'  
POINTS_DIR = 'img0_points'
OUT_COUNTS = 'meta/img0_tile_counts.csv'         
OUT_POINTS = 'meta/img0_points_fullimage.csv'    

os.makedirs(POINTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_COUNTS), exist_ok=True)


df_idx = pd.read_csv(INDEX_CSV).sort_values('tile_id')
assert {'x','y','tile_id','filename'}.issubset(df_idx.columns), "INDEX_CSV have to contain: tile_id, filename, x, y"
rows = list(df_idx.to_dict('records'))

# 
done = {os.path.splitext(f)[0] for f in os.listdir(POINTS_DIR) if f.endswith('.npy')}
rows = [r for r in rows if os.path.splitext(r['filename'])[0] not in done]

i = 0
rows_out_counts = []   
rows_out_points = [] 

status = W.HTML()
btn_prev   = W.Button(description='◀ Prev')
btn_undo   = W.Button(description='Undo')
btn_skip   = W.Button(description='Skip ⏭')
btn_finish = W.Button(description='Finish', button_style='success')

fig, ax = plt.subplots(figsize=(6,6))
scat = ax.scatter([], [], s=24)
pts = []
cid = None

def save_all():
    """make csv"""
    # ---- save----
    if rows_out_counts:
        new_c = pd.DataFrame(rows_out_counts)
        if os.path.exists(OUT_COUNTS):
            old_c = pd.read_csv(OUT_COUNTS)
            # overwrite
            keep_mask = ~old_c['tile_id'].isin(new_c['tile_id'])
            out_c = pd.concat([old_c[keep_mask], new_c], ignore_index=True).sort_values('tile_id')
        else:
            out_c = new_c.sort_values('tile_id')
        out_c.to_csv(OUT_COUNTS, index=False)

    # ---- save----
    if rows_out_points:
        new_p = pd.DataFrame(rows_out_points)
        if os.path.exists(OUT_POINTS):
            old_p = pd.read_csv(OUT_POINTS)
            keep_mask = ~old_p['tile_id'].isin(new_p['tile_id'])
            out_p = pd.concat([old_p[keep_mask], new_p], ignore_index=True)
        else:
            out_p = new_p
        # 
        out_p = out_p.sort_values(['tile_id','pid'])
        out_p.to_csv(OUT_POINTS, index=False)

def load_tile(k):
    """load slides"""
    global pts, cid, scat
    ax.clear()
    if k >= len(rows):
        save_all()
        status.value = f"<b>finished！</b> wrote <code>{OUT_COUNTS}</code> and <code>{OUT_POINTS}</code>"
        fig.canvas.draw_idle()
        for b in (btn_prev, btn_undo, btn_skip, btn_finish): b.disabled = True
        return
    r = rows[k]
    img = np.array(Image.open(os.path.join(TILES_DIR, r['filename'])))
    ax.imshow(img); ax.set_axis_off()
    pts = []
    scat = ax.scatter([], [], s=24)

    def onclick(event):
        if event.inaxes is not ax: return
        if event.button == 1:
            pts.append((event.xdata, event.ydata))
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            scat.set_offsets(np.c_[xs, ys])
            fig.canvas.draw_idle()

    global cid
    if cid is not None:
        fig.canvas.mpl_disconnect(cid)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    status.value = f"Tile {k+1}/{len(rows)} | id={r['tile_id']} | {r['filename']} | 已点 <b>{len(pts)}</b>"

def on_prev(_):
    global i
    if i > 0:
        i -= 1
        load_tile(i)

def on_undo(_):
    global scat
    if pts:
        pts.pop()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        scat.remove()
        scat = ax.scatter(xs, ys, s=24)
        fig.canvas.draw_idle()
        # update
        parts = status.value.split('|')
        parts[-1] = f"schon <b>{len(pts)}</b>"
        status.value = '|'.join(parts)

def on_skip(_):
    global i
    i += 1
    load_tile(i)

def on_finish(_):
    """save, nect slide"""
    global i
    r = rows[i]
    stem = os.path.splitext(r['filename'])[0]

   
    np.save(os.path.join(POINTS_DIR, stem + '.npy'), np.array(pts, dtype=float))

  
    rows_out_counts.append({
        'tile_id': int(r['tile_id']),
        'filename': r['filename'],
        'count_gt': int(len(pts))
    })

   
    
    tx, ty = float(r['x']), float(r['y'])
   
    for pid, (xl, yl) in enumerate(pts):
        x_crop = tx + float(xl)
        y_crop = ty + float(yl)
        x_full = CROP_X0 + x_crop
        y_full = CROP_Y0 + y_crop
        rows_out_points.append({
            'tile_id': int(r['tile_id']),
            'filename': r['filename'],
            'pid': int(pid),           
            'x_local': float(xl), 'y_local': float(yl),
            'x_crop':  float(x_crop),  'y_crop':  float(y_crop),
            'x_full':  float(x_full),  'y_full':  float(y_full),
        })

  
    save_all()

  
    i += 1
    load_tile(i)

btn_prev.on_click(on_prev)
btn_undo.on_click(on_undo)
btn_skip.on_click(on_skip)
btn_finish.on_click(on_finish)

ui = W.VBox([status, W.HBox([btn_prev, btn_undo, btn_skip, btn_finish])])
display(ui, fig.canvas)

load_tile(i)


get_ipython().run_line_magic('matplotlib', 'widget')


import ipywidgets as W
from IPython.display import display


CROP_X0 = 7000   
CROP_Y0 = 17500   

# ---------- path ----------
TILES_DIR = 'tiles/img1'
INDEX_CSV = 'meta/img1_tile_index.csv' 
POINTS_DIR = 'img1_points'
OUT_COUNTS = 'meta/img1_tile_counts.csv'          
OUT_POINTS = 'meta/img1_points_fullimage.csv'     

os.makedirs(POINTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_COUNTS), exist_ok=True)


df_idx = pd.read_csv(INDEX_CSV).sort_values('tile_id')
assert {'x','y','tile_id','filename'}.issubset(df_idx.columns), "INDEX_CSV have to contain: tile_id, filename, x, y"
rows = list(df_idx.to_dict('records'))


done = {os.path.splitext(f)[0] for f in os.listdir(POINTS_DIR) if f.endswith('.npy')}
rows = [r for r in rows if os.path.splitext(r['filename'])[0] not in done]

i = 0
rows_out_counts = [] 
rows_out_points = []  

status = W.HTML()
btn_prev   = W.Button(description='◀ Prev')
btn_undo   = W.Button(description='Undo')
btn_skip   = W.Button(description='Skip ⏭')
btn_finish = W.Button(description='Finish ✅', button_style='success')

fig, ax = plt.subplots(figsize=(6,6))
scat = ax.scatter([], [], s=24)
pts = []
cid = None

def save_all():
    """save and overwrite"""
  
    if rows_out_counts:
        new_c = pd.DataFrame(rows_out_counts)
        if os.path.exists(OUT_COUNTS):
            old_c = pd.read_csv(OUT_COUNTS)
           
            keep_mask = ~old_c['tile_id'].isin(new_c['tile_id'])
            out_c = pd.concat([old_c[keep_mask], new_c], ignore_index=True).sort_values('tile_id')
        else:
            out_c = new_c.sort_values('tile_id')
        out_c.to_csv(OUT_COUNTS, index=False)

   
    if rows_out_points:
        new_p = pd.DataFrame(rows_out_points)
        if os.path.exists(OUT_POINTS):
            old_p = pd.read_csv(OUT_POINTS)
            keep_mask = ~old_p['tile_id'].isin(new_p['tile_id'])
            out_p = pd.concat([old_p[keep_mask], new_p], ignore_index=True)
        else:
            out_p = new_p
       
        out_p = out_p.sort_values(['tile_id','pid'])
        out_p.to_csv(OUT_POINTS, index=False)

def load_tile(k):
    """load k slide"""
    global pts, cid, scat
    ax.clear()
    if k >= len(rows):
        save_all()
        status.value = f"<b>done！</b> geschriebt <code>{OUT_COUNTS}</code> und <code>{OUT_POINTS}</code>"
        fig.canvas.draw_idle()
        for b in (btn_prev, btn_undo, btn_skip, btn_finish): b.disabled = True
        return
    r = rows[k]
    img = np.array(Image.open(os.path.join(TILES_DIR, r['filename'])))
    ax.imshow(img); ax.set_axis_off()
    pts = []
    scat = ax.scatter([], [], s=24)

    def onclick(event):
        if event.inaxes is not ax: return
        if event.button == 1:
            pts.append((event.xdata, event.ydata))
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            scat.set_offsets(np.c_[xs, ys])
            fig.canvas.draw_idle()

    global cid
    if cid is not None:
        fig.canvas.mpl_disconnect(cid)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    status.value = f"Tile {k+1}/{len(rows)} | id={r['tile_id']} | {r['filename']} | schon <b>{len(pts)}</b>"

def on_prev(_):
    global i
    if i > 0:
        i -= 1
        load_tile(i)

def on_undo(_):
    global scat
    if pts:
        pts.pop()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        scat.remove()
        scat = ax.scatter(xs, ys, s=24)
        fig.canvas.draw_idle()
       
        parts = status.value.split('|')
        parts[-1] = f" 已点 <b>{len(pts)}</b>"
        status.value = '|'.join(parts)

def on_skip(_):
    global i
    i += 1
    load_tile(i)

def on_finish(_):
    """保存该 tile 的 .npy、计数和逐点全局坐标，然后进入下一张"""
    global i
    r = rows[i]
    stem = os.path.splitext(r['filename'])[0]

   
    np.save(os.path.join(POINTS_DIR, stem + '.npy'), np.array(pts, dtype=float))

   
    rows_out_counts.append({
        'tile_id': int(r['tile_id']),
        'filename': r['filename'],
        'count_gt': int(len(pts))
    })

  
    tx, ty = float(r['x']), float(r['y'])
  
    for pid, (xl, yl) in enumerate(pts):
        x_crop = tx + float(xl)
        y_crop = ty + float(yl)
        x_full = CROP_X0 + x_crop
        y_full = CROP_Y0 + y_crop
        rows_out_points.append({
            'tile_id': int(r['tile_id']),
            'filename': r['filename'],
            'pid': int(pid),           # 该 tile 内点的序号
            'x_local': float(xl), 'y_local': float(yl),
            'x_crop':  float(x_crop),  'y_crop':  float(y_crop),
            'x_full':  float(x_full),  'y_full':  float(y_full),
        })

   
    save_all()

   
    i += 1
    load_tile(i)

btn_prev.on_click(on_prev)
btn_undo.on_click(on_undo)
btn_skip.on_click(on_skip)
btn_finish.on_click(on_finish)

ui = W.VBox([status, W.HBox([btn_prev, btn_undo, btn_skip, btn_finish])])
display(ui, fig.canvas)

load_tile(i)


# In[ ]:





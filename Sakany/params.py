#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# Settings
# ========
# dim             : dimension, number of features for position
# attr            : considered attribute (name in "networkx" representation), excluding position
# attr_ncomp      : number of components for `attr`
# attr_label_list : list of name(s) for plots
# attr_cmap_list  : list of color map(s) to use for plots 
# -----
dim = 3
attr = 'log10_csdim'
attr_ncomp =  2
attr_label_list = ['log10_cw', 'log10_ch']
attr_cmap_list = ['plasma', 'viridis']
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Parameters for plot
# ===================

# figure size
figsize_equal_def = (6, 6)

figsize_equal_l1 = ( 7,  7)
figsize_equal_l2 = ( 8,  8)
figsize_equal_l3 = ( 9,  9)
figsize_equal_l4 = (10, 10)
figsize_equal_l5 = (11, 11)
figsize_equal_l6 = (12, 12)

figsize_h1 = (6, 5)
figsize_h2 = (6, 4)
figsize_h3 = (6, 3)
figsize_h4 = (6, 2)
figsize_h5 = (6, 1)

figsize_lh1 = (9, 7.5)
figsize_lh2 = (9, 6  )
figsize_lh3 = (9, 4.5)
figsize_lh4 = (9, 3  )
figsize_lh5 = (9, 1.5)

# Plot graph in 2d
# ################

figsize_single  = [6, 6] # single graph in 2d
figsize_multi   = [9, 9] # multi graphs in 2d
figsize_multi_s = [9, 9] # multi graphs (smaller) in 2d
figsize_multi_line = [9, 2] # multi graphs (in line) in 2d

# Dictionaries (keyword arguments) for plotting graphs in 2d
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2D view - single (default)
# --------------------------
kwds_single = {
    'node_shape':'o',         # node symbol
    'edgecolors':'black',     # node border color
    'node_size':32,           # node size
    'linewidths':0.2,         # node border width
    'edge_color':'black',     # edge color
    'width':1.0,              # edge width
    'font_size':9,            # font size (for labels)
    'font_color':'black',     # font color (for labels)
    'node_color':'tab:blue',  # node color (no attr)
}

# 2D view - mutliple graphs
# -------------------------
kwds_multi = {
    'node_shape':'o',         # node symbol
    'edgecolors':'black',     # node border color
    'node_size':18,           # node size
    'linewidths':0.2,         # node border width
    'edge_color':'black',     # edge color
    'width':1.0,              # edge width
    'font_size':9,            # font size (for labels)
    'font_color':'black',     # font color (for labels)
    'node_color':'tab:blue',  # node color (no attr)
}

# 2D view - mutliple graphs (smaller)
# -----------------------------------
kwds_multi_s = {
    'node_shape':'o',         # node symbol
    'edgecolors':'black',     # node border color
    'node_size':15,           # node size
    'linewidths':0.2,         # node border width
    'edge_color':'black',     # edge color
    'width':.5,               # edge width
    'font_size':9,            # font size (for labels)
    'font_color':'black',     # font color (for labels)
    'node_color':'tab:blue',  # node color (no attr)
}

# 2D view - mutliple graphs (in line)
# -----------------------------------
kwds_multi_line = {
    'node_shape':'o',         # node symbol
    'edgecolors':'black',     # node border color
    'node_size':10,           # node size
    'linewidths':0.2,         # node border width
    'edge_color':'black',     # edge color
    'width':.4,               # edge width
    'font_size':9,            # font size (for labels)
    'font_color':'black',     # font color (for labels)
    'node_color':'tab:blue',  # node color (no attr)
}

# Plot graph in 3d (pyvista)
# ##########################

window_size_single  = [1024, 768]  # single graph in 3d
window_size_multi   = [1500, 900]  # multi graphs in 3d
window_size_multi_s = [1500, 900]  # multi graphs (smaller) in 3d
window_size_multi_line = [1500, 200]  # multi graphs (in line) in 3d

# Dictionaries (keyword arguments) for plotting graphs in 3d
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 3D view - single graph
# ----------------------
kwargs_edges_single = {
    'line_width':1.2,  # edge width
    'color':'black', # edge color 
    'opacity':.8,    # opacity
}
                
kwargs_pts_single = {
    'render_points_as_spheres':True,  # rendering
    'point_size':12,                  # point size
    'color':'tab:blue',               # point color (if attr=None)
}    

kwargs_scalar_bar_single = {
    'vertical':False,
    'title_font_size':16,
    'label_font_size':14,
}

kwargs_pts_labels_single = {
    'point_size':0, # point size
    'font_size':24, # font size
    'text_color':'black',        
}

# 3D view - mutliple graphs
# -------------------------
kwargs_edges_multi = {
    'line_width':1,  # edge width
    'color':'black',   # edge color 
    'opacity':.8,      # opacity
}
                
kwargs_pts_multi = {
    'render_points_as_spheres':True,  # rendering
    'point_size':8,                   # point size
    'color':'tab:blue',               # point color (if attr=None)
}    

kwargs_scalar_bar_multi = {
    'vertical':False,
    'title_font_size':12,
    'label_font_size':10
}

kwargs_pts_labels_multi = {
    'point_size':0, # point size
    'font_size':24, # font size
    'text_color':'black',        
}

# 3D view - mutliple graphs (small)
# ---------------------------------
kwargs_edges_multi_s = {
    'line_width':.8, # edge width
    'color':'black', # edge color 
    'opacity':.8,    # opacity
}
                
kwargs_pts_multi_s = {
    'render_points_as_spheres':True,  # rendering
    'point_size':6,                   # point size
    'color':'tab:blue',               # point color (if attr=None)
}    

kwargs_scalar_bar_multi_s = {
    'vertical':False,
    'title_font_size':10,
    'label_font_size':8
}

kwargs_pts_labels_multi_s = {
    'point_size':0, # point size
    'font_size':24, # font size
    'text_color':'black',        
}

# 3D view - mutliple graphs (in line)
# -----------------------------------
kwargs_edges_multi_line = {
    'line_width':.5,  # edge width
    'color':'black', # edge color 
    'opacity':.8,    # opacity
}
                
kwargs_pts_multi_line = {
    'render_points_as_spheres':True,  # rendering
    'point_size':5,                   # point size
    'color':'tab:blue',               # point color (if attr=None)
}    

kwargs_scalar_bar_multi_line = {
    'vertical':False,
    'title_font_size':8,
    'label_font_size':6
}

kwargs_pts_labels_multi_line = {
    'point_size':0, # point size
    'font_size':24, # font size
    'text_color':'black',        
}

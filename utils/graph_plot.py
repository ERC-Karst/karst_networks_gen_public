#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File (Python):  'graph_plot.py'
author:         Julien Straubhaar
date:           2024

Functions to plot graphs.
"""

import numpy as np
import networkx

import matplotlib.pyplot as plt
import pyvista as pv

import torch, torch_geometric

# =============================================================================
# Utils for plotting graph
# =============================================================================
# -----------------------------------------------------------------------------
def plot_graph_2d(G, pos_attr='pos', attr=None, attr_ind=0, 
                  title=None, title_fontsize=12, 
                  with_labels=False, show_colorbar=True,
                  **kwargs):
    """
    Plot a graph in 2D, in the current axis figure.

    The graph is assumed to have node attribute corresponding to the spatial 
    position.

    Parameters
    ----------
    G : networkx.Graph object
        graph

    pos_attr : str, default: 'pos'
        name of the node attribute corresponding to the position of the nodes 
    
    attr : str, optional
        name of the node attribute to be plotted
    
    attr_ind : int, default: 0
        index of the attribute to be plotted (used if attr is not `None`)
    
    title : str, optional
        title of the figure

    title_fontsize : int, default: 12
        font size used for title

    with_labels : bool, default: False
        indicates if the node id is plotted (`True`) or not (`False`)
    
    show_colorbar : bool, default: True
        indicates if the color bar is displayed (`True`) or not (`False`);
        used if `attr` is not `None`

    kwargs : dict
        additional arguments, possible keys, values : 
        
        - 'node_shape':'o',         # node symbol
        - 'edgecolors':'green',     # node border color
        - 'node_size':100,          # node size
        - 'linewidths':3.2,         # node border width
        - 'edge_color':'pink',      # edge color
        - 'width':3.0,              # edge width
        - 'font_size':9,            # font size (labels)
        - 'font_color':'red',       # font color (labels)
        - 'node_color':'lightblue', # node color (used if `attr=None`)
        - 'cmap':'plasma'           # color map (used if `attr` is not `None`) 
        - 'vmin':vmin               # min value for color map (used if `attr` is not `None`) 
        - 'vmax':vmax               # max value for color map (used if `attr` is not `None`) 
    """
    # fname = 'plot_graph_2d'
    
    kwds = kwargs.copy()

    dict_pos2d = {i: p[:2] for i, p in networkx.get_node_attributes(G, pos_attr).items()}

    if attr is None:
        # remove unappropriate keyword arguments
        for k in ('cmap', 'vmin', 'vmax'):
            if k in kwds.keys():
                del kwds[k]
        networkx.draw(G, pos=dict_pos2d, with_labels=with_labels, **kwds)
    
    else:
        # get value of the attribute (and index) to plot        
        v = np.asarray(list(networkx.get_node_attributes(G, attr).values()))[:, attr_ind]

        # get keyword arguments for edge and remove them from kwds
        if 'edge_color' in kwds.keys():
            edge_color = kwds['edge_color']
            del kwds['edge_color']
        else:
            edge_color = None

        if 'width' in kwds.keys():
            width = kwds['width']
            del kwds['width']
        else:
            width = None

        _ = networkx.draw_networkx_edges(G, pos=dict_pos2d, edge_color=edge_color, width=width)

        # get keyword arguments for labels and remove them from kwds
        if 'font_color' in kwds.keys():
            font_color = kwds['font_color']
            del kwds['font_color']
        else:
            font_color = None

        if 'font_size' in kwds.keys():
            font_size = kwds['font_size']
            del kwds['font_size']
        else:
            font_size = None

        # remove unappropriate keyword arguments
        for k in ('node_color',):
            if k in kwds.keys():
                del kwds[k]

        im_nodes = networkx.draw_networkx_nodes(G, pos=dict_pos2d, node_color=list(v), **kwds)

        if with_labels:
            for ni in G.nodes():
                plt.text(*G.nodes[ni][pos_attr][:2], str(ni), 
                        horizontalalignment='center',
                        verticalalignment='center',
                        color=font_color, fontsize=font_size
                    )
        
        if show_colorbar:
            plt.colorbar(im_nodes)

        if title is not None:
            plt.title(title, font_size=title_fontsize)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def plot_graph_3d(G, pos_attr='pos', attr=None, attr_ind=0, attr_label=None, 
                  plotter=None,
                  title=None, title_fontsize=12,
                  with_labels=False, show_scalar_bar=True, 
                  kwargs_edges=None, kwargs_pts=None, kwargs_scalar_bar=None, kwargs_pts_labels=None, 
                  cpos=None, print_cpos=False):
    """
    Plot a graph in 3D (based on `pyvista`).

    The graph is assumed to node attribute corresponding to the spatial 
    position.

    Parameters
    ----------
    G : networkx.Graph object
        graph

    pos_attr : str, default: 'pos'
        name of the node attribute corresponding to the position of the nodes 

    attr : str, optional
        name of the node attribute to be plotted

    attr_ind : int, default: 0
        index of the attribute to be plotted (used if attr is not `None`)
        
    attr_label : str, optional
        label for the colorbar, used if `attr` is not `None`;
        by default (`None`): `attr_label` is set to `attr`_`attr_ind`

    plotter : :class:`pyvista.Plotter`, optional
        - if given (not `None`), add element to the plotter, a further call to \
        `plotter.show()` will be required to show the plot
        - if not given (`None`, default): a plotter is created and the plot \
        is shown
    
    title : str, optional
        title of the figure

    title_fontsize : int, default: 12
        font size used for title

    with_labels : bool, default: False
        indicates if the node id is plotted (`True`) or not (`False`)

    show_scalar_bar : bool, default: True
        indicates if the scalar bar is displayed (`True`) or not (`False`);
        used if `attr` is not `None`

    kwargs_edges : dict, optional
        keyword arguments passed to function `plotter.add_mesh` to plot edges,
        possible keys, values : 
        
        - 'line_width':3,  # edge width
        - 'color':'black', # edge color 
        - 'opacity':.8,    # opacity
        
    kwargs_pts : dict, optional
        keyword arguments passed to function `plotter.add_mesh` to plot nodes (points),
        possible keys, values :         
        
        - 'render_points_as_spheres':True,  # rendering
        - 'point_size':10,                  # point size
        - 'color':'black',                  # point color (if attr=None)
        - 'cmap':'plasma',                  # color map (if attr is not None)
        - 'clim':[vmin, vmax],              # min and max for color map (if attr is not None)
    
    kwargs_pts_labels : dict, optional
        keyword arguments passed to function `plotter.add_point_labels` to plot labels
        (used if `with_labels=True`), possible keys, values : 
        
        - 'point_size':0, # point size
        - 'font_size':24, # font size
        - 'color':'black',
    
    kwargs_scalar_bar : dict, optional
        keyword arguments passed to function `plotter.add_scalar_bar` to plot scalar bar
        (used if `attr` is not None), possible keys, values : 
        
        - 'vertical':True,
        - 'title_font_size':24,
        - 'label_font_size':12,

    cpos : sequence[sequence[float]], optional
        camera position (unsused if `plotter=None`);
        `cpos` = [camera_location, focus_point, viewup_vector], with

        - camera_location: (tuple of length 3) camera location ("eye")
        - focus_point    : (tuple of length 3) focus point
        - viewup_vector  : (tuple of length 3) viewup vector (vector \
        attached to the "head" and pointed to the "sky")

        note: in principle, (focus_point - camera_location) is orthogonal to
        viewup_vector

    print_cpos : bool, default: False
        indicates if camera position is printed (in stdout); used only if 
        `plotter` is `None`
    """
    # fname = 'plot_graph_3d'

    mesh = get_mesh_from_graph(G, pos_attr=pos_attr)

    if kwargs_edges is None:
        kwargs_edges = {}

    if kwargs_pts is None:
        kwargs_pts = {}

    if kwargs_scalar_bar is None:
        kwargs_scalar_bar = {}

    if kwargs_pts_labels is None:
        kwargs_pts_labels = {}

    if plotter is not None:
        pp = plotter
    else:
        pp = pv.Plotter()

    pp.add_mesh(mesh, **kwargs_edges)           # draw edges
    
    if attr is None:
        pp.add_mesh(mesh.points, **kwargs_pts)  # draw nodes
    else:
        # get value of the attribute (and index) to plot        
        v = np.asarray(list(networkx.get_node_attributes(G, attr).values()))[:, attr_ind]
        
        if attr_label is None:
            attr_label = f'{attr}_{attr_ind}'

        # Set points with color
        pts = pv.PolyData(mesh.points)
        pts[attr_label] = v # colors for the points

        if 'cmap' in kwargs_pts.keys():
            if 'color' in kwargs_pts.keys():
                del kwargs_pts['color']

        if 'title' not in kwargs_scalar_bar:
            scalar_bar_title_added = True
            kwargs_scalar_bar['title'] = attr_label
        else:
            scalar_bar_title_added = False

        pp.add_mesh(pts, **kwargs_pts, show_scalar_bar=False)    # draw nodes
        if show_scalar_bar:
            pp.add_scalar_bar(**kwargs_scalar_bar)               # add scalar bar (color bar)

        if scalar_bar_title_added:
            del kwargs_scalar_bar['title']

    if with_labels:
        pp.add_point_labels(mesh.points, np.arange(mesh.n_points), **kwargs_pts_labels) # add point labels
    
    if title is not None:
        pp.add_text(title, font_size=title_fontsize)

    if plotter is None:
        pp.camera_position = cpos # set camera position
        cpos = pp.show(return_cpos=True)
        if print_cpos: 
            print(cpos)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def plot_graph_single_2d_from_G_networkx(
        G, out_name='', pos_attr='pos', attr=None,
        attr_label_list=None, attr_cmap_list=None,
        title=None, title_fontsize=12,
        figsize=None, 
        save_fig_png=False, filename_prefix='./',
        with_labels=False, show_color_bar=True,
        show=True,
        **kwargs):
    """
    Generates plot(s) of single graph in 2D.

    One figure per component of the attribute (`attr`) is generated (at least
    one figure).

    The graph is assumed to have node attribute corresponding to the spatial 
    position.

    Parameters:
    ----------
    G : `networkx.Graph`
        graph
    
    out_name : str, default: ''
        string (should not contain space) used in default title, and in filename

    pos_attr : str
        name of the node attribute corresponding to the position of the nodes 
    
    attr : str, optional
        name of the node attribute to be plotted
    
    attr_label_list : list of strs, optional
        name of each component of attribute `attr`;
        used if `attr` is not `None`
    
    attr_cmap_list : list of color map, optional
        color map used for ach component of `attr`;
        used if `attr` is not `None`

    title : str, optional
        title of the figure; 
        by default (`None`): `out_name` is used in title

    title_fontsize : int, default: 12
        font size used for title

    figsize : 2-tuple of ints or floats, optional
        figure size in inches

    save_fig_png : bool, default: False
        indicates if the figure(s) is(are) saved in a png file

    filename_prefix: str, default: './'
        beginning of the name of the file(s) in which the figure(s)
        are saved (used if `save_fig_png=True`)

    with_labels : bool, default: False
        indicates if the node id is plotted (`True`) or not (`False`)
       
    show_color_bar : bool, default: True
        indicates if the color bar is displayed (`True`) or not (`False`);
        used if `attr` is not `None`

    show : bool, default: True
        indicates if the graphics is shown with `matplotlib.pyplot.show()`

    kwargs : dict
        additional arguments, possible keys, values : 
        
        - 'node_shape':'o',         # node symbol
        - 'edgecolors':'green',     # node border color
        - 'node_size':100,          # node size
        - 'linewidths':3.2,         # node border width
        - 'edge_color':'pink',      # edge color
        - 'width':3.0,              # edge width
        - 'font_size':9,            # font size (labels)
        - 'font_color':'red',       # font color (labels)
        - 'node_color':'lightblue', # node color (used if `attr=None`)
    """
    kwds = kwargs.copy()
    
    if attr is not None:
        # === with attribute(s) ===
        attr_ncomp = len(G.nodes[0][attr])
        if attr_label_list is None:
            attr_label_list = [f'{attr}_{i_attr}' for i_attr in range(attr_ncomp)]
            attr_cmap_list = attr_ncomp*['viridis']

        for i_attr in range(attr_ncomp):
            # loop on attributes
            attr_label = attr_label_list[i_attr]
            attr_cmap = attr_cmap_list[i_attr]
            
            kwds['cmap'] = attr_cmap

            # Plot
            # ----
            plt.figure(figsize=figsize)
            plot_graph_2d(
                    G, pos_attr=pos_attr, attr=attr, attr_ind=i_attr, 
                    with_labels=with_labels, show_colorbar=show_color_bar, 
                    **kwds)
            # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            # plt.axis('on')
            plt.axis('equal')
            if title is not None:
                plt.title(title, fontsize=title_fontsize)
            else:
                plt.title(f'{out_name} - {attr_label}, n_nodes={G.number_of_nodes()}', fontsize=title_fontsize)

            if save_fig_png:
                plt.tight_layout()
                plt.savefig(f'{filename_prefix}_{out_name}_{attr}_{i_attr}_2d.png')

            if show:
                plt.show()
            else:
                plt.close()

    else:
        # === no attribute ===
        # Plot
        # ----
        plt.figure(figsize=figsize)
        plot_graph_2d(G, pos_attr=pos_attr, attr=attr, with_labels=with_labels, **kwds)
        # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # plt.axis('on')
        plt.axis('equal')
        if title is not None:
            plt.title(title, fontsize=title_fontsize)
        else:
            plt.title(f'{out_name}, n_nodes={G.number_of_nodes()}', fontsize=title_fontsize)

        if save_fig_png:
            plt.tight_layout()
            plt.savefig(f'{filename_prefix}_{out_name}_2d.png')

        if show:
            plt.show()
        else:
            plt.close()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def plot_graph_single_3d_from_G_networkx(
        G, out_name='', pos_attr='pos', attr=None,
        attr_label_list=None, attr_cmap_list=None,
        title=None, title_fontsize=12,
        notebook=True, window_size=[1024, 768], off_screen=False,
        save_fig_png=False, filename_prefix='./',
        with_labels=False, show_color_bar=True,
        kwargs_edges=None, kwargs_pts=None, kwargs_scalar_bar=None, kwargs_pts_labels=None, 
        cpos=None, print_cpos=False):
    """
    Generates plot(s) of single graph in 3D.

    One figure per component of the attribute (`attr`) is generated (at least
    one figure).

    The graph is assumed to have node attribute corresponding to the spatial 
    position.

    Parameters:
    ----------
    G : `networkx.Graph`
        graph
    
    out_name : str, default: ''
        string (should not contain space) used in default title, and in filename

    pos_attr : str, default: 'pos'
        name of the node attribute corresponding to the position of the nodes 

    attr : str, optional
        name of the node attribute to be plotted

    attr_label_list : list of strs, optional
        name of each component of attribute `attr`;
        used if `attr` is not `None`
    
    attr_cmap_list : list of color map, optional
        color map used for ach component of `attr`;
        used if `attr` is not `None`

    title : str, optional
        title of the figure; 
        by default (`None`): `out_name` is used in title

    title_fontsize : int, default: 12
        font size used for title
    
    notebook : bool, default: True
        notebook mode for pyvista Plotter
        - True: in line figure
        - False: figures in pop-up window

    window_size : sequence of 2 ints, default: [1024, 768]
        figure size in pixels

    off_screen : bool, default: False
        off_screen mode for pyvista Plotter

    save_fig_png : bool, default: False
        indicates if the figure(s) is(are) saved in a png file

    filename_prefix: str, default: './'
        beginning of the name of the file(s) in which the figure(s)
        are saved (used if `save_fig_png=True`)

    with_labels : bool, default: False
        indicates if the node id is plotted (`True`) or not (`False`)

    show_color_bar : bool, default: True
        indicates if the scalar bar is displayed (`True`) or not (`False`);
        used if `attr` is not `None`

    kwargs_edges : dict, optional
        keyword arguments passed to function `plotter.add_mesh` to plot edges,
        possible keys, values : 
        
        - 'line_width':3,  # edge width
        - 'color':'black', # edge color 
        - 'opacity':.8,    # opacity
        
    kwargs_pts : dict, optional
        keyword arguments passed to function `plotter.add_mesh` to plot nodes (points),
        possible keys, values :         
        
        - 'render_points_as_spheres':True,  # rendering
        - 'point_size':10,                  # point size
        - 'color':'black',                  # point color (if attr=None)
    
    kwargs_pts_labels : dict, optional
        keyword arguments passed to function `plotter.add_point_labels` to plot labels
        (used if `with_labels=True`), possible keys, values : 
        
        - 'point_size':0, # point size
        - 'font_size':24, # font size
        - 'color':'black',
    
    kwargs_scalar_bar : dict, optional
        keyword arguments passed to function `plotter.add_scalar_bar` to plot scalar bar
        (used if `attr` is not None), possible keys, values : 
        
        - 'vertical':True,
        - 'title_font_size':24,
        - 'label_font_size':12,

    cpos : sequence[sequence[float]], optional
        camera position (unsused if `plotter=None`);
        `cpos` = [camera_location, focus_point, viewup_vector], with

        - camera_location: (tuple of length 3) camera location ("eye")
        - focus_point    : (tuple of length 3) focus point
        - viewup_vector  : (tuple of length 3) viewup vector (vector \
        attached to the "head" and pointed to the "sky")

        note: in principle, (focus_point - camera_location) is orthogonal to
        viewup_vector

    print_cpos : bool, default: False
        indicates if camera position is printed (in stdout)
    """
    kwds_edges      = kwargs_edges.copy()      if kwargs_edges      is not None else {}
    kwds_pts        = kwargs_pts.copy()        if kwargs_pts        is not None else {}
    kwds_pts_labels = kwargs_pts_labels.copy() if kwargs_pts_labels is not None else {}
    kwds_scalar_bar = kwargs_scalar_bar.copy() if kwargs_scalar_bar is not None else {}

    if attr is not None:
        # === with attribute(s) ===
        attr_ncomp = len(G.nodes[0][attr])
        if attr_label_list is None:
            attr_label_list = [f'{attr}_{i_attr}' for i_attr in range(attr_ncomp)]
            attr_cmap_list = attr_ncomp*['viridis']

        for i_attr in range(attr_ncomp):
            # loop on attributes
            attr_label = attr_label_list[i_attr]
            attr_cmap = attr_cmap_list[i_attr]

            kwds_pts['cmap'] = attr_cmap
            kwds_scalar_bar['title'] = attr_label

            # Plot
            # ----
            pp = pv.Plotter(notebook=notebook, window_size=window_size, off_screen=off_screen)
            plot_graph_3d(
                    G, pos_attr=pos_attr, attr=attr, attr_ind=i_attr, attr_label=attr_label,
                    plotter=pp, with_labels=with_labels, show_scalar_bar=show_color_bar,
                    kwargs_edges=kwds_edges, kwargs_pts=kwds_pts, kwargs_scalar_bar=kwds_scalar_bar, kwargs_pts_labels=kwds_pts_labels)
            if title is not None:
                pp.add_text(title, font_size=title_fontsize)
            else:
                pp.add_text(f'{out_name} - {attr_label}, n_nodes={G.number_of_nodes()}', font_size=title_fontsize)

            pp.add_bounding_box()
            # pp.show_bounds()
            pp.show_axes()

            pp.camera_position = cpos # set camera position

            if save_fig_png and notebook:
                pp.screenshot(f'{filename_prefix}_{out_name}_{attr}_{i_attr}_3d.png', transparent_background=False)

            if not off_screen:
                cpos = pp.show(return_cpos=True)
                if print_cpos:
                    print(cpos)
            else:
                pp.close()
                
    else:
        # === no attribute ===
        # Plot
        # ----
        pp = pv.Plotter(notebook=notebook, window_size=window_size, off_screen=off_screen)
        plot_graph_3d(
                G, pos_attr=pos_attr, attr=attr,
                plotter=pp, with_labels=with_labels,
                kwargs_edges=kwds_edges, kwargs_pts=kwds_pts, kwargs_pts_labels=kwds_pts_labels)
        if title is not None:
            pp.add_text(title, font_size=title_fontsize)
        else:
            pp.add_text(f'{out_name}, n_nodes={G.number_of_nodes()}', font_size=title_fontsize)
        
        pp.add_bounding_box()
        # pp.show_bounds()
        pp.show_axes()

        pp.camera_position = cpos # set camera position

        if save_fig_png and notebook:
            pp.screenshot(f'{filename_prefix}_{out_name}_3d.png', transparent_background=False)

        if not off_screen:
            cpos = pp.show(return_cpos=True)
            if print_cpos:
                print(cpos)
        else:
            pp.close()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def plot_graph_multi_2d_from_G_networkx_list(
        G_list, out_name='', nr=None, pos_attr='pos', attr=None,
        attr_label_list=None, attr_cmap_list=None,
        title_list=None, title_fontsize=12,
        figsize=None, 
        save_fig_png=False, filename_prefix='./',
        with_labels=False, same_color_bar=False, show_color_bar=True,
        show=True,
        **kwargs):
    """
    Generates plot(s) of multiple graphs in 2D.

    One figure per component of the attribute (`attr`) is generated (at least
    one figure), each figure is divided in sub-figure for each graph in the
    given list.

    The graph is assumed to have node attribute corresponding to the spatial 
    position.

    Parameters:
    ----------
    G_list : list of `networkx.Graph`
        list of graphs

    out_name : str, default: ''
        string (should not contain space) used in sup-title, and in filename
    
    nr : int, optional
        number of rows in the figure;
        by default (`None`) : set to `int(np.sqrt(len(G_list)))`

    pos_attr : str
        name of the node attribute corresponding to the position of the nodes 
    
    attr : str, optional
        name of the node attribute to be plotted
    
    attr_label_list : list of strs, optional
        name of each component of attribute `attr`;
        used if `attr` is not `None`
    
    attr_cmap_list : list of color map, optional
        color map used for ach component of `attr`;
        used if `attr` is not `None`

    title_list : str, optional
        list of titles for each subplot

    title_fontsize : int, default: 12
        font size used for title

    figsize : 2-tuple of ints or floats, optional
        figure size in inches

    save_fig_png : bool, default: False
        indicates if the figure(s) is(are) saved in a png file

    filename_prefix: str, default: './'
        beginning of the name of the file(s) in which the figure(s)
        are saved (used if `save_fig_png=True`)

    with_labels : bool, default: False
        indicates if the node id is plotted (`True`) or not (`False`)
       
    same_color_bar : bool, default: False
        indicates if same color bar is used for all subplots (used only
        if attribute is plotted)

    show_color_bar : bool, default: True
        indicates if the color bar is displayed (`True`) or not (`False`);
        used if `attr` is not `None`

    show : bool, default: True
        indicates if the graphics is shown with `matplotlib.pyplot.show()`

    kwargs : dict
        additional arguments, possible keys, values : 
        
        - 'node_shape':'o',         # node symbol
        - 'edgecolors':'green',     # node border color
        - 'node_size':100,          # node size
        - 'linewidths':3.2,         # node border width
        - 'edge_color':'pink',      # edge color
        - 'width':3.0,              # edge width
        - 'font_size':9,            # font size (labels)
        - 'font_color':'red',       # font color (labels)
        - 'node_color':'lightblue', # node color (used if `attr=None`)
    """
    kwds = kwargs.copy()

    ng = len(G_list)
    if nr is None:
        nr = int(np.sqrt(ng))
    nc = ng//nr + (ng%nr>0)

    if attr is not None:
        # === with attribute(s) ===
        attr_ncomp = len(G_list[0].nodes[0][attr])
        if attr_label_list is None:
            attr_label_list = [f'{attr}_{i_attr}' for i_attr in range(attr_ncomp)]
            attr_cmap_list = attr_ncomp*['viridis']

        show_color_bar_loc = show_color_bar
        if same_color_bar:
            # min max of attributes
            v = np.vstack([np.asarray(list(networkx.get_node_attributes(G, attr).values())) for G in G_list])
            vmin_list, vmax_list = v.min(axis=0), v.max(axis=0)

        for i_attr in range(attr_ncomp):
            # loop on attributes
            attr_label = attr_label_list[i_attr]
            attr_cmap = attr_cmap_list[i_attr]

            kwds['cmap'] = attr_cmap
            if same_color_bar:
                kwds['vmin'] = vmin_list[i_attr]
                kwds['vmax'] = vmax_list[i_attr]
            else:
                for k in ('vmin', 'vmax'):
                    if k in kwds.keys():
                        del kwds[k]

            # Plot
            # ----
            plt.subplots(nr, nc, figsize=figsize)
            for i, G in enumerate(G_list):
                plt.subplot(nr, nc, i+1)
                if same_color_bar and show_color_bar:
                    show_color_bar_loc = i==ng-1
                plot_graph_2d(
                        G, pos_attr=pos_attr, attr=attr, attr_ind=i_attr, 
                        with_labels=with_labels, show_colorbar=show_color_bar_loc, 
                        **kwds)
                # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
                # plt.axis('on')
                plt.axis('equal')
                if title_list is not None:
                    plt.title(title_list[i], fontsize=title_fontsize)
                else:
                    plt.title(f'n_nodes={G.number_of_nodes()}', fontsize=title_fontsize)

            for i in range(ng, nr*nc):
                plt.subplot(nr, nc, i+1)
                plt.axis('off')

            plt.suptitle(f'{out_name} - {attr_label}')

            if save_fig_png:
                plt.tight_layout()
                plt.savefig(f'{filename_prefix}_{out_name}_{attr}_{i_attr}_2d.png')

            if show:
                plt.show()
            else:
                plt.close()

    else:
        # === no attribute ===
        # Plot
        # ----
        plt.subplots(nr, nc, figsize=figsize)
        for i, G in enumerate(G_list):
            plt.subplot(nr, nc, i+1)
            plot_graph_2d(G, pos_attr=pos_attr, attr=attr, with_labels=with_labels, **kwds)
            # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            # plt.axis('on')
            plt.axis('equal')
            if title_list is not None:
                plt.title(title_list[i], fontsize=title_fontsize)
            else:
                plt.title(f'n_nodes={G.number_of_nodes()}', fontsize=title_fontsize)

        for i in range(ng, nr*nc):
            plt.subplot(nr, nc, i+1)
            plt.axis('off')

        plt.suptitle(f'{out_name}')

        if save_fig_png:
            plt.tight_layout()
            plt.savefig(f'{filename_prefix}_{out_name}_2d.png')

        if show:
            plt.show()
        else:
            plt.close()
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def plot_graph_multi_3d_from_G_networkx_list(
        G_list, out_name='', nr=None, pos_attr='pos', attr=None,
        attr_label_list=None, attr_cmap_list=None,
        title_list=None, title_fontsize=12,
        notebook=True, window_size=[1024, 768], off_screen=False,
        save_fig_png=False, filename_prefix='./',
        with_labels=False, same_color_bar=False, show_color_bar=True,
        kwargs_edges=None, kwargs_pts=None, kwargs_scalar_bar=None, kwargs_pts_labels=None, 
        cpos=None, print_cpos=False):
    """
    Generates plot(s) of multiple graphs in 3D.

    One figure per component of the attribute (`attr`) is generated (at least
    one figure), each figure is divided in sub-figure for each graph in the
    given list.

    The graph is assumed to have node attribute corresponding to the spatial 
    position.
    
    Parameters:
    ----------
    G_list : list of `networkx.Graph`
        list of graphs

    out_name : str, default: ''
        string (should not contain space) used in default title, and in filename
    
    nr : int, optional
        number of rows in the figure;
        by default (`None`) : set to `int(np.sqrt(len(G_list)))`

    pos_attr : str, default: 'pos'
        name of the node attribute corresponding to the position of the nodes 

    attr : str, optional
        name of the node attribute to be plotted

    attr_label_list : list of strs, optional
        name of each component of attribute `attr`;
        used if `attr` is not `None`
    
    attr_cmap_list : list of color map, optional
        color map used for ach component of `attr`;
        used if `attr` is not `None`

    title_list : str, optional
        list of titles for each subplot

    title_fontsize : int, default: 12
        font size used for title
    
    notebook : bool, default: True
        notebook mode for pyvista Plotter
        - True: in line figure
        - False: figures in pop-up window

    window_size : sequence of 2 ints, default: [1024, 768]
        figure size in pixels

    off_screen : bool, default: False
        off_screen mode for pyvista Plotter

    save_fig_png : bool, default: False
        indicates if the figure(s) is(are) saved in a png file

    filename_prefix: str, default: './'
        beginning of the name of the file(s) in which the figure(s)
        are saved (used if `save_fig_png=True`)

    with_labels : bool, default: False
        indicates if the node id is plotted (`True`) or not (`False`)

    same_color_bar : bool, default: False
        indicates if same color bar is used for all subplots (used only
        if attribute is plotted)

    show_color_bar : bool, default: True
        indicates if the scalar bar is displayed (`True`) or not (`False`);
        used if `attr` is not `None`

    kwargs_edges : dict, optional
        keyword arguments passed to function `plotter.add_mesh` to plot edges,
        possible keys, values : 
        
        - 'line_width':3,  # edge width
        - 'color':'black', # edge color 
        - 'opacity':.8,    # opacity
        
    kwargs_pts : dict, optional
        keyword arguments passed to function `plotter.add_mesh` to plot nodes (points),
        possible keys, values :         
        
        - 'render_points_as_spheres':True,  # rendering
        - 'point_size':10,                  # point size
        - 'color':'black',                  # point color (if attr=None)
    
    kwargs_pts_labels : dict, optional
        keyword arguments passed to function `plotter.add_point_labels` to plot labels
        (used if `with_labels=True`), possible keys, values : 
        
        - 'point_size':0, # point size
        - 'font_size':24, # font size
        - 'color':'black',
    
    kwargs_scalar_bar : dict, optional
        keyword arguments passed to function `plotter.add_scalar_bar` to plot scalar bar
        (used if `attr` is not None), possible keys, values : 
        
        - 'vertical':True,
        - 'title_font_size':24,
        - 'label_font_size':12,

    cpos : sequence[sequence[float]], optional
        camera position (unsused if `plotter=None`);
        `cpos` = [camera_location, focus_point, viewup_vector], with

        - camera_location: (tuple of length 3) camera location ("eye")
        - focus_point    : (tuple of length 3) focus point
        - viewup_vector  : (tuple of length 3) viewup vector (vector \
        attached to the "head" and pointed to the "sky")

        note: in principle, (focus_point - camera_location) is orthogonal to
        viewup_vector

    print_cpos : bool, default: False
        indicates if camera position is printed (in stdout)
    """
    kwds_edges      = kwargs_edges.copy()      if kwargs_edges      is not None else {}
    kwds_pts        = kwargs_pts.copy()        if kwargs_pts        is not None else {}
    kwds_pts_labels = kwargs_pts_labels.copy() if kwargs_pts_labels is not None else {}
    kwds_scalar_bar = kwargs_scalar_bar.copy() if kwargs_scalar_bar is not None else {}

    ng = len(G_list)
    if nr is None:
        nr = int(np.sqrt(ng))
    nc = ng//nr + (ng%nr>0)

    if attr is not None:
        # === with attribute(s) ===
        attr_ncomp = len(G_list[0].nodes[0][attr])
        if attr_label_list is None:
            attr_label_list = [f'{attr}_{i_attr}' for i_attr in range(attr_ncomp)]
            attr_cmap_list = attr_ncomp*['viridis']

        if same_color_bar:
            # min max of attributes
            v = np.vstack([np.asarray(list(networkx.get_node_attributes(G, attr).values())) for G in G_list])
            vmin_list, vmax_list = v.min(axis=0), v.max(axis=0)

        for i_attr in range(attr_ncomp):
            # loop on attributes
            attr_label = attr_label_list[i_attr]
            attr_cmap = attr_cmap_list[i_attr]

            kwds_pts['cmap'] = attr_cmap
            if same_color_bar:
                kwds_pts['clim'] = [vmin_list[i_attr], vmax_list[i_attr]]
                kwds_scalar_bar['title'] = ' ' # same for all
            else:
                if 'clim' in kwds_pts.keys():
                    del kwds_pts['clim']
                if 'title' in kwds_scalar_bar.keys():
                    del kwds_scalar_bar['title']

            # Plot
            # ----
            pp = pv.Plotter(notebook=notebook, window_size=window_size, shape=(nr, nc), off_screen=off_screen)
            for i, G in enumerate(G_list):
                pp.subplot(i//nc, i%nc)
                if not same_color_bar:
                    kwds_scalar_bar['title'] = i*' '
                plot_graph_3d(
                        G, pos_attr=pos_attr, attr=attr, attr_ind=i_attr, attr_label=attr_label,
                        plotter=pp, with_labels=with_labels, show_scalar_bar=show_color_bar,
                        kwargs_edges=kwds_edges, kwargs_pts=kwds_pts, kwargs_scalar_bar=kwds_scalar_bar, kwargs_pts_labels=kwds_pts_labels)
                if title_list is not None:
                    pp.add_text(title_list[i], font_size=title_fontsize)
                else:
                    if i==0:
                        pp.add_text(f'n_nodes={G.number_of_nodes()}\n[{out_name} - {attr_label}]', font_size=title_fontsize)
                    else:
                        pp.add_text(f'n_nodes={G.number_of_nodes()}', font_size=title_fontsize)
                
                pp.add_bounding_box()
                # pp.show_bounds()
                pp.show_axes()

            # pp.link_views()
            pp.camera_position = cpos # set camera position

            if save_fig_png and notebook:
                pp.screenshot(f'{filename_prefix}_{out_name}_{attr}_{i_attr}_3d.png', transparent_background=False)

            if not off_screen:
                cpos = pp.show(return_cpos=True)
                if print_cpos:
                    print(cpos)
            else:
                pp.close()

    else:
        # === no attribute ===
        # Plot
        # ----
        pp = pv.Plotter(notebook=notebook, window_size=window_size, shape=(nr, nc), off_screen=off_screen)
        for i, G in enumerate(G_list):
            pp.subplot(i//nc, i%nc)
            plot_graph_3d(
                    G, pos_attr=pos_attr, attr=attr,
                    plotter=pp, with_labels=with_labels,
                    kwargs_edges=kwds_edges, kwargs_pts=kwds_pts, kwargs_pts_labels=kwds_pts_labels)
            if title_list is not None:
                pp.add_text(title_list[i], font_size=title_fontsize)
            else:
                if i==0:
                    pp.add_text(f'n_nodes={G.number_of_nodes()}\n[{out_name}]', font_size=title_fontsize)
                else:
                    pp.add_text(f'n_nodes={G.number_of_nodes()}', font_size=title_fontsize)

            pp.add_bounding_box()
            # pp.show_bounds()
            pp.show_axes()

        # pp.link_views()
        pp.camera_position = cpos # set camera position

        if save_fig_png and notebook:
            pp.screenshot(f'{filename_prefix}_{out_name}_3d.png', transparent_background=False)

        if not off_screen:
            cpos = pp.show(return_cpos=True)
            if print_cpos:
                print(cpos)
        else:
            pp.close()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def plot_graph_single_2d_from_G_geom(
        G_geom, dim, out_name='', attr=None,
        attr_label_list=None, attr_cmap_list=None,
        rescale=False, correction_first=False, node_features_shift_inv=None, node_features_scale_factor_inv=None,
        title=None, title_fontsize=12,
        figsize=None, 
        save_fig_png=False, filename_prefix='./',
        with_labels=False, show_color_bar=True,
        show=True,
        **kwargs):
    """
    Generates plot(s) of single graph in 2D.

    The attribute named 'x' must be used and must contain the spatial position of 
    the nodes as the first `dim` components. One figure per component of the 'x' 
    attribute after the spatial position is generated (at least one figure).

    Parameters:
    ----------
    G_geom : `torch_geometric.data.Data`
        graph (torch_geometric)
    
    dim : int
        spatial dimension (2 or 3) of the graph nodes

    out_name : str, default: ''
        string (should not contain space) used in default title, and in filename
    
    attr : str, optional
        name of the node attribute to be plotted (stored in 'x', after
        the spatial position)

    attr_label_list : list of strs, optional
        name of each component of attribute `attr`;
        used if `attr` is not `None`
    
    attr_cmap_list : list of color map, optional
        color map used for ach component of `attr`;
        used if `attr` is not `None`

    rescale : bool, default: False
        - True: features (including position) are transformed by applying \
        "scaling factor inv", then "shift inv" 
        - False: features are not transformed
    
    correction_first : bool, default: False
        used if `rescale=True`: 
        
        - `True`: the features of each graph are first corrected (centralized by \
        substracting the mean), then transformed (see `rescale`)
        - `False`: the features are directly transformed (see `rescale`)

        Note: for a graph generated by ddpm, the features values tends to 
        be shifted, then the correction should be applied.

    node_features_shift_inv : sequence of floats, optional
        shift inverse, for transforming features, 
        list of same length as the number of attributes in `attr` 
        (including positions); 
        note: must be specified if `rescale=True` (unused otherwise)

    node_features_scale_factor_inv : sequence of floats, optional
        scaling factor inverse, for transforming features, 
        list of same length as the number of attributes in `attr` 
        (including positions); 
        note: must be specified if `rescale=True` (unused otherwise)

    title : str, optional
        title of the figure; 
        by default (`None`): `out_name` is used in title

    title_fontsize : int, default: 12
        font size used for title

    figsize : 2-tuple of ints or floats, optional
        figure size in inches

    save_fig_png : bool, default: False
        indicates if the figure(s) is(are) saved in a png file

    filename_prefix: str, default: './'
        beginning of the name of the file(s) in which the figure(s)
        are saved (used if `save_fig_png=True`)

    with_labels : bool, default: False
        indicates if the node id is plotted (`True`) or not (`False`)
       
    show_color_bar : bool, default: True
        indicates if the color bar is displayed (`True`) or not (`False`);
        used if `attr` is not `None`

    show : bool, default: True
        indicates if the graphics is shown with `matplotlib.pyplot.show()`

    kwargs : dict
        additional arguments, possible keys, values : 
        
        - 'node_shape':'o',         # node symbol
        - 'edgecolors':'green',     # node border color
        - 'node_size':100,          # node size
        - 'linewidths':3.2,         # node border width
        - 'edge_color':'pink',      # edge color
        - 'width':3.0,              # edge width
        - 'font_size':9,            # font size (labels)
        - 'font_color':'red',       # font color (labels)
        - 'node_color':'lightblue', # node color (used if `attr=None`)
    """
    kwds = kwargs.copy()

    G_geom_c = G_geom.clone() # work on clone, because operation (if rescale=True) are inplace...
    
    if rescale:
        # change type
        node_features_shift_inv = torch.from_numpy(np.asarray(node_features_shift_inv)).to(torch.float)
        node_features_scale_factor_inv = torch.from_numpy(np.asarray(node_features_scale_factor_inv)).to(torch.float)

        # transform features
        if rescale:
            if correction_first:
                G_geom_c.x = node_features_shift_inv + node_features_scale_factor_inv*(G_geom_c.x - torch.mean(G_geom_c.x, dim=0))
            else:
                G_geom_c.x = node_features_shift_inv + node_features_scale_factor_inv*G_geom_c.x

    if attr is not None:
        # === with attribute(s) ===
        attr_ncomp = G_geom_c.x.shape[1] - dim
        if attr_label_list is None:
            attr_label_list = [f'{attr}_{i_attr}' for i_attr in range(attr_ncomp)]
            attr_cmap_list = attr_ncomp*['viridis']

        for i_attr in range(attr_ncomp):
            # loop on attributes
            attr_label = attr_label_list[i_attr]
            attr_cmap = attr_cmap_list[i_attr]

            kwds['cmap'] = attr_cmap

            # Plot
            # ----
            G = torch_geometric.utils.to_networkx(G_geom_c, to_undirected=True)
            x = G_geom_c.x.numpy().astype('float')

            dict_pos2d = {i: xi[:2] for i, xi in enumerate(x)}
            networkx.set_node_attributes(G, dict_pos2d, 'pos')

            dict_v = {i: xi[i_attr+dim:i_attr+dim+1] for i, xi in enumerate(x)}
            networkx.set_node_attributes(G, dict_v, attr_label)

            plt.figure(figsize=figsize)
            plot_graph_2d(
                    G, pos_attr='pos', attr=attr_label, attr_ind=0, 
                    with_labels=with_labels, show_colorbar=show_color_bar, 
                    **kwds)
            # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            # plt.axis('on')
            plt.axis('equal')
            if title is not None:
                plt.title(title, fontsize=title_fontsize)
            else:
                plt.title(f'{out_name} - {attr_label}, n_nodes={G.number_of_nodes()}', fontsize=title_fontsize)

            if save_fig_png:
                plt.tight_layout()
                plt.savefig(f'{filename_prefix}_{out_name}_{attr}_{i_attr}_2d.png')

            if show:
                plt.show()
            else:
                plt.close()

    else:
        # === no attribute ===
        # Plot
        # ----
        G = torch_geometric.utils.to_networkx(G_geom_c, to_undirected=True)
        x = G_geom_c.x.numpy().astype('float')

        dict_pos2d = {i: xi[:2] for i, xi in enumerate(x)}
        networkx.set_node_attributes(G, dict_pos2d, 'pos')

        plt.figure(figsize=figsize)
        plot_graph_2d(G, pos_attr='pos', attr=attr, with_labels=with_labels, **kwds)
        # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # plt.axis('on')
        plt.axis('equal')
        if title is not None:
            plt.title(title, fontsize=title_fontsize)
        else:
            plt.title(f'{out_name}, n_nodes={G.number_of_nodes()}', fontsize=title_fontsize)

        if save_fig_png:
            plt.tight_layout()
            plt.savefig(f'{filename_prefix}_{out_name}_2d.png')

        if show:
            plt.show()
        else:
            plt.close()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def plot_graph_single_3d_from_G_geom(
        G_geom, dim, out_name='', attr=None,
        attr_label_list=None, attr_cmap_list=None,
        rescale=False, correction_first=False, node_features_shift_inv=None, node_features_scale_factor_inv=None,
        title=None, title_fontsize=12,
        notebook=True, window_size=[1024, 768], off_screen=False,
        save_fig_png=False, filename_prefix='./',
        with_labels=False, show_color_bar=True,
        kwargs_edges=None, kwargs_pts=None, kwargs_scalar_bar=None, kwargs_pts_labels=None, 
        cpos=None, print_cpos=False):
    """
    Generates plot(s) of multiple graphs in 3D.

    The attribute named 'x' must be used and must contain the spatial position of 
    the nodes as the first `dim` components. One figure per component of the 'x' 
    attribute after the spatial position is generated (at least one figure).

    Parameters:
    ----------
    G_geom_list : list of `torch_geometric.data.Data`
        list of graphs (torch_geometric)

    dim : int
        spatial dimension (should be 3) of the graph nodes

    out_name : str, default: ''
        string (should not contain space) used in default title, and in filename

    attr : str, optional
        name of the node attribute to be plotted (stored in 'x', after
        the spatial position)

    attr_label_list : list of strs, optional
        name of each component of attribute `attr`;
        used if `attr` is not `None`
    
    attr_cmap_list : list of color map, optional
        color map used for ach component of `attr`;
        used if `attr` is not `None`

    rescale : bool, default: False
        - True: features (including position) are transformed by applying \
        "scaling factor inv", then "shift inv" 
        - False: features are not transformed
    
    correction_first : bool, default: False
        used if `rescale=True`: 
        
        - `True`: the features of each graph are first corrected (centralized by \
        substracting the mean), then transformed (see `rescale`)
        - `False`: the features are directly transformed (see `rescale`)

        Note: for a graph generated by ddpm, the features values tends to 
        be shifted, then the correction should be applied.

    node_features_shift_inv : sequence of floats, optional
        shift inverse, for transforming features, 
        list of same length as the number of attributes in `attr` 
        (including positions); 
        note: must be specified if `rescale=True` (unused otherwise)

    node_features_scale_factor_inv : sequence of floats, optional
        scaling factor inverse, for transforming features, 
        list of same length as the number of attributes in `attr` 
        (including positions); 
        note: must be specified if `rescale=True` (unused otherwise)

    title : str, optional
        title of the figure; 
        by default (`None`): `out_name` is used in title

    title_fontsize : int, default: 12
        font size used for title
    
    notebook : bool, default: True
        notebook mode for pyvista Plotter
        - True: in line figure
        - False: figures in pop-up window

    window_size : sequence of 2 ints, default: [1024, 768]
        figure size in pixels

    off_screen : bool, default: False
        off_screen mode for pyvista Plotter

    save_fig_png : bool, default: False
        indicates if the figure(s) is(are) saved in a png file

    filename_prefix: str, default: './'
        beginning of the name of the file(s) in which the figure(s)
        are saved (used if `save_fig_png=True`)

    with_labels : bool, default: False
        indicates if the node id is plotted (`True`) or not (`False`)

    show_color_bar : bool, default: True
        indicates if the scalar bar is displayed (`True`) or not (`False`);
        used if `attr` is not `None`

    kwargs_edges : dict, optional
        keyword arguments passed to function `plotter.add_mesh` to plot edges,
        possible keys, values : 
        
        - 'line_width':3,  # edge width
        - 'color':'black', # edge color 
        - 'opacity':.8,    # opacity
        
    kwargs_pts : dict, optional
        keyword arguments passed to function `plotter.add_mesh` to plot nodes (points),
        possible keys, values :         
        
        - 'render_points_as_spheres':True,  # rendering
        - 'point_size':10,                  # point size
        - 'color':'black',                  # point color (if attr=None)
    
    kwargs_pts_labels : dict, optional
        keyword arguments passed to function `plotter.add_point_labels` to plot labels
        (used if `with_labels=True`), possible keys, values : 
        
        - 'point_size':0, # point size
        - 'font_size':24, # font size
        - 'color':'black',
    
    kwargs_scalar_bar : dict, optional
        keyword arguments passed to function `plotter.add_scalar_bar` to plot scalar bar
        (used if `attr` is not None), possible keys, values : 
        
        - 'vertical':True,
        - 'title_font_size':24,
        - 'label_font_size':12,

    cpos : sequence[sequence[float]], optional
        camera position (unsused if `plotter=None`);
        `cpos` = [camera_location, focus_point, viewup_vector], with

        - camera_location: (tuple of length 3) camera location ("eye")
        - focus_point    : (tuple of length 3) focus point
        - viewup_vector  : (tuple of length 3) viewup vector (vector \
        attached to the "head" and pointed to the "sky")

        note: in principle, (focus_point - camera_location) is orthogonal to
        viewup_vector

    print_cpos : bool, default: False
        indicates if camera position is printed (in stdout)
    """
    kwds_edges      = kwargs_edges.copy()      if kwargs_edges      is not None else {}
    kwds_pts        = kwargs_pts.copy()        if kwargs_pts        is not None else {}
    kwds_pts_labels = kwargs_pts_labels.copy() if kwargs_pts_labels is not None else {}
    kwds_scalar_bar = kwargs_scalar_bar.copy() if kwargs_scalar_bar is not None else {}

    G_geom_c = G_geom.clone() # work on clone, because operation (if rescale=True) are inplace...

    if rescale:
        # change type
        node_features_shift_inv = torch.from_numpy(np.asarray(node_features_shift_inv)).to(torch.float)
        node_features_scale_factor_inv = torch.from_numpy(np.asarray(node_features_scale_factor_inv)).to(torch.float)

        # transform features
        if rescale:
            if correction_first:
                G_geom_c.x = node_features_shift_inv + node_features_scale_factor_inv*(G_geom_c.x - torch.mean(G_geom_c.x, dim=0))
            else:
                G_geom_c.x = node_features_shift_inv + node_features_scale_factor_inv*G_geom_c.x

    if attr is not None:
        # === with attribute(s) ===
        attr_ncomp = G_geom_c.x.shape[1] - dim
        if attr_label_list is None:
            attr_label_list = [f'{attr}_{i_attr}' for i_attr in range(attr_ncomp)]
            attr_cmap_list = attr_ncomp*['viridis']

        for i_attr in range(attr_ncomp):
            # loop on attributes
            attr_label = attr_label_list[i_attr]
            attr_cmap = attr_cmap_list[i_attr]

            kwds_pts['cmap'] = attr_cmap

            # Plot
            # ----
            G = torch_geometric.utils.to_networkx(G_geom_c, to_undirected=True)
            x = G_geom_c.x.numpy().astype('float')

            dict_pos = {i: xi[:3] for i, xi in enumerate(x)}
            networkx.set_node_attributes(G, dict_pos, 'pos')

            dict_v = {i: xi[i_attr+dim:i_attr+dim+1] for i, xi in enumerate(x)}
            networkx.set_node_attributes(G, dict_v, attr_label)

            pp = pv.Plotter(notebook=notebook, window_size=window_size, off_screen=off_screen)
            plot_graph_3d(
                    G, pos_attr='pos', attr=attr_label, attr_ind=0, attr_label=attr_label, 
                    plotter=pp, with_labels=with_labels, show_scalar_bar=show_color_bar,
                    kwargs_edges=kwds_edges, kwargs_pts=kwds_pts, kwargs_scalar_bar=kwds_scalar_bar, kwargs_pts_labels=kwds_pts_labels)
            if title is not None:
                pp.add_text(title, font_size=title_fontsize)
            else:
                pp.add_text(f'n_nodes={G.number_of_nodes()}\n[{out_name} - {attr_label}]', font_size=title_fontsize)

            pp.add_bounding_box()
            # pp.show_bounds()
            pp.show_axes()

            pp.camera_position = cpos # set camera position

            if save_fig_png and notebook:
                pp.screenshot(f'{filename_prefix}_{out_name}_{attr}_{i_attr}_3d.png', transparent_background=False)

            if not off_screen:
                cpos = pp.show(return_cpos=True)
                if print_cpos:
                    print(cpos)
            else:
                pp.close()

    else:
        # === no attribute ===
        # Plot
        # ----
        G = torch_geometric.utils.to_networkx(G_geom_c, to_undirected=True)
        x = G_geom_c.x.numpy().astype('float')

        dict_pos = {i: xi[:3] for i, xi in enumerate(x)}
        networkx.set_node_attributes(G, dict_pos, 'pos')

        pp = pv.Plotter(notebook=notebook, window_size=window_size, off_screen=off_screen)
        plot_graph_3d(
                G, pos_attr='pos', attr=attr,
                plotter=pp, with_labels=with_labels,
                kwargs_edges=kwds_edges, kwargs_pts=kwds_pts, kwargs_pts_labels=kwds_pts_labels)
        if title is not None:
            pp.add_text(title, font_size=title_fontsize)
        else:
            pp.add_text(f'n_nodes={G.number_of_nodes()}\n[{out_name}]', font_size=title_fontsize)

        pp.add_bounding_box()
        # pp.show_bounds()
        pp.show_axes()

        pp.camera_position = cpos # set camera position

        if save_fig_png and notebook:
            pp.screenshot(f'{filename_prefix}_{out_name}_3d.png', transparent_background=False)

        if not off_screen:
            cpos = pp.show(return_cpos=True)
            if print_cpos:
                print(cpos)
        else:
            pp.close()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def plot_graph_multi_2d_from_G_geom_list(
        G_geom_list, dim, out_name='', nr=None, attr=None,
        attr_label_list=None, attr_cmap_list=None,
        rescale=False, correction_first=False, node_features_shift_inv=None, node_features_scale_factor_inv=None,
        title_list=None, title_fontsize=12,
        figsize=None, 
        save_fig_png=False, filename_prefix='./',
        with_labels=False, same_color_bar=False, show_color_bar=True,
        show=True,
        **kwargs):
    """
    Generates plot(s) of multiple graphs in 2D.

    The attribute named 'x' must be used and must contain the spatial position of 
    the nodes as the first `dim` components. One figure per component of the 'x' 
    attribute after the spatial position is generated (at least one figure), each 
    figure is divided in sub-figure for each graph in the given list.

    Parameters:
    ----------
    G_geom_list : list of `torch_geometric.data.Data`
        list of graphs (torch_geometric)
    
    dim : int
        spatial dimension (2 or 3) of the graph nodes

    out_name : str, default: ''
        string (should not contain space) used in sup-title, and in filename
    
    nr : int, optional
        number of rows in the figure;
        by default (`None`) : set to `int(np.sqrt(len(G_list)))`
   
    attr : str, optional
        name of the node attribute to be plotted (stored in 'x', after
        the spatial position)

    attr_label_list : list of strs, optional
        name of each component of attribute `attr`;
        used if `attr` is not `None`
    
    attr_cmap_list : list of color map, optional
        color map used for ach component of `attr`;
        used if `attr` is not `None`

    rescale : bool, default: False
        - True: features (including position) are transformed by applying \
        "scaling factor inv", then "shift inv" 
        - False: features are not transformed
    
    correction_first : bool, default: False
        used if `rescale=True`: 
        
        - `True`: the features of each graph are first corrected (centralized by \
        substracting the mean), then transformed (see `rescale`)
        - `False`: the features are directly transformed (see `rescale`)

        Note: for a graph generated by ddpm, the features values tends to 
        be shifted, then the correction should be applied.

    node_features_shift_inv : sequence of floats, optional
        shift inverse, for transforming features, 
        list of same length as the number of attributes in `attr` 
        (including positions); 
        note: must be specified if `rescale=True` (unused otherwise)

    node_features_scale_factor_inv : sequence of floats, optional
        scaling factor inverse, for transforming features, 
        list of same length as the number of attributes in `attr` 
        (including positions); 
        note: must be specified if `rescale=True` (unused otherwise)

    title_list : str, optional
        list of titles for each subplot

    title_fontsize : int, default: 12
        font size used for title

    figsize : 2-tuple of ints or floats, optional
        figure size in inches

    save_fig_png : bool, default: False
        indicates if the figure(s) is(are) saved in a png file

    filename_prefix: str, default: './'
        beginning of the name of the file(s) in which the figure(s)
        are saved (used if `save_fig_png=True`)

    with_labels : bool, default: False
        indicates if the node id is plotted (`True`) or not (`False`)
       
    same_color_bar : bool, default: False
        indicates if same color bar is used for all subplots (used only
        if attribute is plotted)

    show_color_bar : bool, default: True
        indicates if the color bar is displayed (`True`) or not (`False`);
        used if `attr` is not `None`

    show : bool, default: True
        indicates if the graphics is shown with `matplotlib.pyplot.show()`

    kwargs : dict
        additional arguments, possible keys, values : 
        
        - 'node_shape':'o',         # node symbol
        - 'edgecolors':'green',     # node border color
        - 'node_size':100,          # node size
        - 'linewidths':3.2,         # node border width
        - 'edge_color':'pink',      # edge color
        - 'width':3.0,              # edge width
        - 'font_size':9,            # font size (labels)
        - 'font_color':'red',       # font color (labels)
        - 'node_color':'lightblue', # node color (used if `attr=None`)
    """
    kwds = kwargs.copy()

    ng = len(G_geom_list)
    if nr is None:
        nr = int(np.sqrt(ng))
    nc = ng//nr + (ng%nr>0)

    # get batch of graph (G_batch.x are features of all graphs)
    G_batch = torch_geometric.data.Batch.from_data_list(G_geom_list) # torch_geometric.data.batch.DataBatch
    if rescale:
        # change type
        node_features_shift_inv = torch.from_numpy(np.asarray(node_features_shift_inv)).to(torch.float)
        node_features_scale_factor_inv = torch.from_numpy(np.asarray(node_features_scale_factor_inv)).to(torch.float)

        # transform features
        if rescale:
            if correction_first:
                for k in range(ng):
                    G_batch.x[G_batch.ptr[k]:G_batch.ptr[k+1]] = node_features_shift_inv + node_features_scale_factor_inv*(G_batch.x[G_batch.ptr[k]:G_batch.ptr[k+1]] - torch.mean(G_batch.x[G_batch.ptr[k]:G_batch.ptr[k+1]], dim=0))
            else:
                G_batch.x = node_features_shift_inv + node_features_scale_factor_inv*G_batch.x

    if attr is not None:
        # === with attribute(s) ===
        attr_ncomp = G_geom_list[0].x.shape[1] - dim
        if attr_label_list is None:
            attr_label_list = [f'{attr}_{i_attr}' for i_attr in range(attr_ncomp)]
            attr_cmap_list = attr_ncomp*['viridis']

        show_color_bar_loc = show_color_bar
        if same_color_bar:
            # min max of attributes
            v = G_batch.x.numpy().astype('float')[:, dim:]
            vmin_list, vmax_list = v.min(axis=0), v.max(axis=0)

        for i_attr in range(attr_ncomp):
            # loop on attributes
            attr_label = attr_label_list[i_attr]
            attr_cmap = attr_cmap_list[i_attr]

            kwds['cmap'] = attr_cmap
            if same_color_bar:
                kwds['vmin'] = vmin_list[i_attr]
                kwds['vmax'] = vmax_list[i_attr]
            else:
                for k in ('vmin', 'vmax'):
                    if k in kwds.keys():
                        del kwds[k]

            # Plot
            # ----
            plt.subplots(nr, nc, figsize=figsize)
            for j, G_geom in enumerate(G_geom_list):
                G = torch_geometric.utils.to_networkx(G_geom, to_undirected=True)
                x = G_batch.x[G_batch.ptr[j]:G_batch.ptr[j+1]].numpy().astype('float')

                dict_pos2d = {i: xi[:2] for i, xi in enumerate(x)}
                networkx.set_node_attributes(G, dict_pos2d, 'pos')

                dict_v = {i: xi[i_attr+dim:i_attr+dim+1] for i, xi in enumerate(x)}
                networkx.set_node_attributes(G, dict_v, attr_label)

                plt.subplot(nr, nc, j+1)
                if same_color_bar and show_color_bar:
                    show_color_bar_loc = j==ng-1
                plot_graph_2d(
                        G, pos_attr='pos', attr=attr_label, attr_ind=0, 
                        with_labels=with_labels, show_colorbar=show_color_bar_loc, 
                        **kwds)
                # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
                # plt.axis('on')
                plt.axis('equal')
                if title_list is not None:
                    plt.title(title_list[j], fontsize=title_fontsize)
                else:
                    plt.title(f'n_nodes={G.number_of_nodes()}', fontsize=title_fontsize)

            for j in range(ng, nr*nc):
                plt.subplot(nr, nc, j+1)
                plt.axis('off')

            plt.suptitle(f'{out_name} - {attr_label}')

            if save_fig_png:
                plt.tight_layout()
                plt.savefig(f'{filename_prefix}_{out_name}_{attr}_{i_attr}_2d.png')

            if show:
                plt.show()
            else:
                plt.close()

    else:
        # === no attribute ===
        # Plot
        # ----
        plt.subplots(nr, nc, figsize=figsize)
        for j, G_geom in enumerate(G_geom_list):
            G = torch_geometric.utils.to_networkx(G_geom, to_undirected=True)
            x = G_batch.x[G_batch.ptr[j]:G_batch.ptr[j+1]].numpy().astype('float')

            dict_pos2d = {i: xi[:2] for i, xi in enumerate(x)}
            networkx.set_node_attributes(G, dict_pos2d, 'pos')

            plt.subplot(nr, nc, j+1)
            plot_graph_2d(G, pos_attr='pos', attr=attr, with_labels=with_labels, **kwds)
            # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            # plt.axis('on')
            plt.axis('equal')
            if title_list is not None:
                plt.title(title_list[j], fontsize=title_fontsize)
            else:
                plt.title(f'n_nodes={G.number_of_nodes()}', fontsize=title_fontsize)

        for j in range(ng, nr*nc):
            plt.subplot(nr, nc, j+1)
            plt.axis('off')

        plt.suptitle(f'{out_name}')

        if save_fig_png:
            plt.tight_layout()
            plt.savefig(f'{filename_prefix}_{out_name}_2d.png')

        if show:
            plt.show()
        else:
            plt.close()
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def plot_graph_multi_3d_from_G_geom_list(
        G_geom_list, dim, out_name='', nr=None, attr=None,
        attr_label_list=None, attr_cmap_list=None,
        rescale=False, correction_first=False, node_features_shift_inv=None, node_features_scale_factor_inv=None,
        title_list=None, title_fontsize=12,
        notebook=True, window_size=[1024, 768], off_screen=False,
        save_fig_png=False, filename_prefix='./',
        with_labels=False, same_color_bar=False, show_color_bar=True,
        kwargs_edges=None, kwargs_pts=None, kwargs_scalar_bar=None, kwargs_pts_labels=None, 
        cpos=None, print_cpos=False):
    """
    Generates plot(s) of multiple graphs in 3D.

    The attribute named 'x' must be used and must contain the spatial position of 
    the nodes as the first `dim` components. One figure per component of the 'x' 
    attribute after the spatial position is generated (at least one figure), each 
    figure is divided in sub-figure for each graph in the given list.
    Generates plot(s) of multiple graphs in 3D.

    Parameters:
    ----------
    G_geom_list : list of `torch_geometric.data.Data`
        list of graphs (torch_geometric)

    dim : int
        spatial dimension (should be 3) of the graph nodes

    out_name : str, default: ''
        string (should not contain space) used in default title, and in filename
    
    nr : int, optional
        number of rows in the figure;
        by default (`None`) : set to `int(np.sqrt(len(G_list)))`

    attr : str, optional
        name of the node attribute to be plotted (stored in 'x', after
        the spatial position)

    attr_label_list : list of strs, optional
        name of each component of attribute `attr`;
        used if `attr` is not `None`
    
    attr_cmap_list : list of color map, optional
        color map used for ach component of `attr`;
        used if `attr` is not `None`

    rescale : bool, default: False
        - True: features (including position) are transformed by applying \
        "scaling factor inv", then "shift inv" 
        - False: features are not transformed
    
    correction_first : bool, default: False
        used if `rescale=True`: 
        
        - `True`: the features of each graph are first corrected (centralized by \
        substracting the mean), then transformed (see `rescale`)
        - `False`: the features are directly transformed (see `rescale`)

        Note: for a graph generated by ddpm, the features values tends to 
        be shifted, then the correction should be applied.

    node_features_shift_inv : sequence of floats, optional
        shift inverse, for transforming features, 
        list of same length as the number of attributes in `attr` 
        (including positions); 
        note: must be specified if `rescale=True` (unused otherwise)

    node_features_scale_factor_inv : sequence of floats, optional
        scaling factor inverse, for transforming features, 
        list of same length as the number of attributes in `attr` 
        (including positions); 
        note: must be specified if `rescale=True` (unused otherwise)

    title_list : str, optional
        list of titles for each subplot

    title_fontsize : int, default: 12
        font size used for title
    
    notebook : bool, default: True
        notebook mode for pyvista Plotter
        - True: in line figure
        - False: figures in pop-up window

    window_size : sequence of 2 ints, default: [1024, 768]
        figure size in pixels

    off_screen : bool, default: False
        off_screen mode for pyvista Plotter

    save_fig_png : bool, default: False
        indicates if the figure(s) is(are) saved in a png file

    filename_prefix: str, default: './'
        beginning of the name of the file(s) in which the figure(s)
        are saved (used if `save_fig_png=True`)

    with_labels : bool, default: False
        indicates if the node id is plotted (`True`) or not (`False`)

    same_color_bar : bool, default: False
        indicates if same color bar is used for all subplots (used only
        if attribute is plotted)

    show_color_bar : bool, default: True
        indicates if the scalar bar is displayed (`True`) or not (`False`);
        used if `attr` is not `None`

    kwargs_edges : dict, optional
        keyword arguments passed to function `plotter.add_mesh` to plot edges,
        possible keys, values : 
        
        - 'line_width':3,  # edge width
        - 'color':'black', # edge color 
        - 'opacity':.8,    # opacity
        
    kwargs_pts : dict, optional
        keyword arguments passed to function `plotter.add_mesh` to plot nodes (points),
        possible keys, values :         
        
        - 'render_points_as_spheres':True,  # rendering
        - 'point_size':10,                  # point size
        - 'color':'black',                  # point color (if attr=None)
    
    kwargs_pts_labels : dict, optional
        keyword arguments passed to function `plotter.add_point_labels` to plot labels
        (used if `with_labels=True`), possible keys, values : 
        
        - 'point_size':0, # point size
        - 'font_size':24, # font size
        - 'color':'black',
    
    kwargs_scalar_bar : dict, optional
        keyword arguments passed to function `plotter.add_scalar_bar` to plot scalar bar
        (used if `attr` is not None), possible keys, values : 
        
        - 'vertical':True,
        - 'title_font_size':24,
        - 'label_font_size':12,

    cpos : sequence[sequence[float]], optional
        camera position (unsused if `plotter=None`);
        `cpos` = [camera_location, focus_point, viewup_vector], with

        - camera_location: (tuple of length 3) camera location ("eye")
        - focus_point    : (tuple of length 3) focus point
        - viewup_vector  : (tuple of length 3) viewup vector (vector \
        attached to the "head" and pointed to the "sky")

        note: in principle, (focus_point - camera_location) is orthogonal to
        viewup_vector

    print_cpos : bool, default: False
        indicates if camera position is printed (in stdout)
    """
    kwds_edges      = kwargs_edges.copy()      if kwargs_edges      is not None else {}
    kwds_pts        = kwargs_pts.copy()        if kwargs_pts        is not None else {}
    kwds_pts_labels = kwargs_pts_labels.copy() if kwargs_pts_labels is not None else {}
    kwds_scalar_bar = kwargs_scalar_bar.copy() if kwargs_scalar_bar is not None else {}

    ng = len(G_geom_list)
    if nr is None:
        nr = int(np.sqrt(ng))
    nc = ng//nr + (ng%nr>0)

    # get batch of graph (G_batch.x are features of all graphs)
    G_batch = torch_geometric.data.Batch.from_data_list(G_geom_list) # torch_geometric.data.batch.DataBatch
    if rescale:
        # change type
        node_features_shift_inv = torch.from_numpy(np.asarray(node_features_shift_inv)).to(torch.float)
        node_features_scale_factor_inv = torch.from_numpy(np.asarray(node_features_scale_factor_inv)).to(torch.float)

        # transform features
        if rescale:
            if correction_first:
                for k in range(ng):
                    G_batch.x[G_batch.ptr[k]:G_batch.ptr[k+1]] = node_features_shift_inv + node_features_scale_factor_inv*(G_batch.x[G_batch.ptr[k]:G_batch.ptr[k+1]] - torch.mean(G_batch.x[G_batch.ptr[k]:G_batch.ptr[k+1]], dim=0))
            else:
                G_batch.x = node_features_shift_inv + node_features_scale_factor_inv*G_batch.x

    if attr is not None:
        # === with attribute(s) ===
        attr_ncomp = G_geom_list[0].x.shape[1] - dim
        if attr_label_list is None:
            attr_label_list = [f'{attr}_{i_attr}' for i_attr in range(attr_ncomp)]
            attr_cmap_list = attr_ncomp*['viridis']

        if same_color_bar:
            # min max of attributes
            v = G_batch.x.numpy().astype('float')[:, dim:]
            vmin_list, vmax_list = v.min(axis=0), v.max(axis=0)

        for i_attr in range(attr_ncomp):
            # loop on attributes
            attr_label = attr_label_list[i_attr]
            attr_cmap = attr_cmap_list[i_attr]

            kwds_pts['cmap'] = attr_cmap
            if same_color_bar:
                kwds_pts['clim'] = [vmin_list[i_attr], vmax_list[i_attr]]
                kwds_scalar_bar['title'] = ' ' # same for all
            else:
                if 'clim' in kwds_pts.keys():
                    del kwds_pts['clim']
                if 'title' in kwds_scalar_bar.keys():
                    del kwds_scalar_bar['title']

            # Plot
            # ----
            pp = pv.Plotter(notebook=notebook, window_size=window_size, shape=(nr, nc), off_screen=off_screen)
            for j, G_geom in enumerate(G_geom_list):
                G = torch_geometric.utils.to_networkx(G_geom, to_undirected=True)
                x = G_batch.x[G_batch.ptr[j]:G_batch.ptr[j+1]].numpy().astype('float')

                dict_pos = {i: xi[:3] for i, xi in enumerate(x)}
                networkx.set_node_attributes(G, dict_pos, 'pos')

                dict_v = {i: xi[i_attr+dim:i_attr+dim+1] for i, xi in enumerate(x)}
                networkx.set_node_attributes(G, dict_v, attr_label)

                pp.subplot(j//nc, j%nc)
                if not same_color_bar:
                    kwds_scalar_bar['title'] = j*' '
                plot_graph_3d(
                        G, pos_attr='pos', attr=attr_label, attr_ind=0, attr_label=attr_label, 
                        plotter=pp, with_labels=with_labels, show_scalar_bar=show_color_bar,
                        kwargs_edges=kwds_edges, kwargs_pts=kwds_pts, kwargs_scalar_bar=kwds_scalar_bar, kwargs_pts_labels=kwds_pts_labels)
                if title_list is not None:
                    pp.add_text(title_list[j], font_size=title_fontsize)
                else:
                    if j==0:
                        pp.add_text(f'n_nodes={G.number_of_nodes()}\n[{out_name} - {attr_label}]', font_size=title_fontsize)
                    else:
                        pp.add_text(f'n_nodes={G.number_of_nodes()}', font_size=title_fontsize)

                pp.add_bounding_box()
                # pp.show_bounds()
                pp.show_axes()

            # pp.link_views()
            pp.camera_position = cpos # set camera position

            if save_fig_png and notebook:
                pp.screenshot(f'{filename_prefix}_{out_name}_{attr}_{i_attr}_3d.png', transparent_background=False)

            if not off_screen:
                cpos = pp.show(return_cpos=True)
                if print_cpos:
                    print(cpos)
            else:
                pp.close()

    else:
        # === no attribute ===
        # Plot
        # ----
        pp = pv.Plotter(notebook=notebook, window_size=window_size, shape=(nr, nc), off_screen=off_screen)
        for j, G_geom in enumerate(G_geom_list):
            G = torch_geometric.utils.to_networkx(G_geom, to_undirected=True)
            x = G_batch.x[G_batch.ptr[j]:G_batch.ptr[j+1]].numpy().astype('float')

            dict_pos = {i: xi[:3] for i, xi in enumerate(x)}
            networkx.set_node_attributes(G, dict_pos, 'pos')

            pp.subplot(j//nc, j%nc)
            plot_graph_3d(
                    G, pos_attr='pos', attr=attr,
                    plotter=pp, with_labels=with_labels,
                    kwargs_edges=kwds_edges, kwargs_pts=kwds_pts, kwargs_pts_labels=kwds_pts_labels)
            if title_list is not None:
                pp.add_text(title_list[j], font_size=title_fontsize)
            else:
                if j==0:
                    pp.add_text(f'n_nodes={G.number_of_nodes()}\n[{out_name}]', font_size=title_fontsize)
                else:
                    pp.add_text(f'n_nodes={G.number_of_nodes()}', font_size=title_fontsize)

            pp.add_bounding_box()
            # pp.show_bounds()
            pp.show_axes()

        # pp.link_views()
        pp.camera_position = cpos # set camera position

        if save_fig_png and notebook:
            pp.screenshot(f'{filename_prefix}_{out_name}_3d.png', transparent_background=False)

        if not off_screen:
            cpos = pp.show(return_cpos=True)
            if print_cpos:
                print(cpos)
        else:
            pp.close()
# -----------------------------------------------------------------------------

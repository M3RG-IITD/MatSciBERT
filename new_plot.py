from functools import partial

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from cycler import cycler
import scipy.stats as stats
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch

plot = plt.plot

plt.savefig = partial(plt.savefig, dpi = 500,bbox_inches = "tight")

def __t(x):
    if plt.rcParams['text.usetex']:
        return r'\textbf{{{}}}'.format(x)
    else:
        return x

def usetex(on='on'):
    if on=='off':
        plt.rcParams['text.usetex'] = False
    elif on=='on':
        plt.rcParams['text.usetex'] = True
    else:
        pass

def set_tick(val='in'):
    params = {
    'xtick.direction': val,
    'ytick.direction': val,
    }
    plt.rcParams.update(params)

def no_box(ax2):
    # Hide the right and top spines
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')

def make_box(ax2):
    # Show the right and top spines
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(True)
    # show ticks on the both spines
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

def set_mood_dark():
    params = {
    'figure.edgecolor': 'white',
    'figure.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.facecolor': 'black',
    'savefig.edgecolor': 'black',
    'savefig.facecolor': 'black',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'boxplot.boxprops.color': 'white',
    'boxplot.capprops.color': 'white',
    'boxplot.flierprops.color': 'white',
    'boxplot.flierprops.markeredgecolor': 'white',
    'boxplot.whiskerprops.color': 'white',
    'grid.color': '#e0e0e0',
    'hatch.color': 'white',
    'image.cmap': 'hot',
    'patch.edgecolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    }
    plt.rcParams.update(params)

def set_mood_light():
    c1 = 'black'
    c2 = 'white'
    params = {
    'figure.edgecolor': c1,
    'figure.facecolor': c2,
    'axes.edgecolor': c1,
    'axes.facecolor': c2,
    'savefig.edgecolor': c1,
    'savefig.facecolor': c2,
    'text.color': c1,
    'axes.labelcolor': c1,
    'boxplot.boxprops.color': c1,
    'boxplot.capprops.color': c1,
    'boxplot.flierprops.color': c1,
    'boxplot.flierprops.markeredgecolor': c1,
    'boxplot.whiskerprops.color': c1,
    'grid.color': '#b0b0b0',
    'hatch.color': c1,
    'image.cmap': 'hot',
    'patch.edgecolor': c1,
    'xtick.color': c1,
    'ytick.color': c1,
    }
    plt.rcParams.update(params)


#family='Arial'
def set_font(size=20,family='Arial',weight='normal'):
    params = {
    'xtick.labelsize': size,
    'ytick.labelsize': size,
    'axes.labelsize': size,
    'axes.titlesize': size,
    'font.size': size,
    'legend.title_fontsize': size,
    'legend.fontsize': size,
    'figure.titlesize': size,
    'font.family': family,
    'font.weight': weight,
    'axes.labelweight':weight,
    'axes.titleweight': weight,
    }
    plt.rcParams.update(params)


Params_ = {'_internal.classic_mode': False,
    'agg.path.chunksize': 0,
    'animation.avconv_args': [],
    'animation.avconv_path': 'avconv',
    'animation.bitrate': -1,
    'animation.codec': 'h264',
    'animation.convert_args': [],
    'animation.convert_path': 'convert',
    'animation.embed_limit': 20.0,
    'animation.ffmpeg_args': [],
    'animation.ffmpeg_path': 'ffmpeg',
    'animation.frame_format': 'png',
    'animation.html': 'none',
    'animation.html_args': [],
    'animation.writer': 'ffmpeg',
    'axes.autolimit_mode': 'data',
    'axes.axisbelow': 'line',
    'axes.edgecolor': 'black',
    'axes.facecolor': 'white',
    'axes.formatter.limits': [-7, 7],
    'axes.formatter.min_exponent': 0,
    'axes.formatter.offset_threshold': 4,
    'axes.formatter.use_locale': False,
    'axes.formatter.use_mathtext': True,
    'axes.formatter.useoffset': True,
    'axes.grid': False,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'both',
    'axes.labelcolor': 'black',
    'axes.labelpad': 4.0,
    'axes.labelsize': 20.0,
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.5,
    'axes.prop_cycle': ((cycler('color', ['blue', 'red', 'green', 'purple', 'brown']) + cycler('linestyle', ['-', '-', '-', '-', '-'])) + cycler('marker', ['', '', '', '', ''])),
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.spines.right': True,
    'axes.spines.top': True,
    'axes.titlepad': 6.0,
    'axes.titlesize': 20.0,
    'axes.titleweight': 'bold',
    'axes.unicode_minus': True,
    'axes.xmargin': 0.05,
    'axes.ymargin': 0.05,
    'axes3d.grid': True,
    'backend': 'module://ipykernel.pylab.backend_inline',
    'backend_fallback': True,
    'boxplot.bootstrap': None,
    'boxplot.boxprops.color': 'black',
    'boxplot.boxprops.linestyle': '-',
    'boxplot.boxprops.linewidth': 1.0,
    'boxplot.capprops.color': 'black',
    'boxplot.capprops.linestyle': '-',
    'boxplot.capprops.linewidth': 1.0,
    'boxplot.flierprops.color': 'black',
    'boxplot.flierprops.linestyle': 'none',
    'boxplot.flierprops.linewidth': 1.0,
    'boxplot.flierprops.marker': 'o',
    'boxplot.flierprops.markeredgecolor': 'black',
    'boxplot.flierprops.markerfacecolor': 'none',
    'boxplot.flierprops.markersize': 10.0,
    'boxplot.meanline': False,
    'boxplot.meanprops.color': 'C2',
    'boxplot.meanprops.linestyle': '--',
    'boxplot.meanprops.linewidth': 1.0,
    'boxplot.meanprops.marker': '^',
    'boxplot.meanprops.markeredgecolor': 'C2',
    'boxplot.meanprops.markerfacecolor': 'C2',
    'boxplot.meanprops.markersize': 10.0,
    'boxplot.medianprops.color': 'C1',
    'boxplot.medianprops.linestyle': '-',
    'boxplot.medianprops.linewidth': 1.0,
    'boxplot.notch': False,
    'boxplot.patchartist': False,
    'boxplot.showbox': True,
    'boxplot.showcaps': True,
    'boxplot.showfliers': True,
    'boxplot.showmeans': False,
    'boxplot.vertical': True,
    'boxplot.whiskerprops.color': 'black',
    'boxplot.whiskerprops.linestyle': '-',
    'boxplot.whiskerprops.linewidth': 1.0,
    'boxplot.whiskers': 1.5,
    'contour.corner_mask': True,
    'contour.negative_linestyle': 'dashed',
    'date.autoformatter.day': '%Y-%m-%d',
    'date.autoformatter.hour': '%m-%d %H',
    'date.autoformatter.microsecond': '%M:%S.%f',
    'date.autoformatter.minute': '%d %H:%M',
    'date.autoformatter.month': '%Y-%m',
    'date.autoformatter.second': '%H:%M:%S',
    'date.autoformatter.year': '%Y',
    'docstring.hardcopy': False,
    'errorbar.capsize': 3.0,
    'figure.autolayout': False,
    'figure.constrained_layout.h_pad': 0.04167,
    'figure.constrained_layout.hspace': 0.02,
    'figure.constrained_layout.use': False,
    'figure.constrained_layout.w_pad': 0.04167,
    'figure.constrained_layout.wspace': 0.02,
    'figure.dpi': 72.0,
    'figure.edgecolor': (1, 1, 1, 0),
    'figure.facecolor': 'w',
    'figure.figsize': [6.0, 6.0],
    'figure.max_open_warning': 20,
    'figure.subplot.bottom': 0.125,
    'figure.subplot.hspace': 0.2,
    'figure.subplot.left': 0.125,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.88,
    'figure.subplot.wspace': 0.2,
    'figure.titlesize': 20.0,
    'figure.titleweight': 'bold',
    'font.cursive': ['Apple Chancery',
                     'Textile',
                     'Zapf Chancery',
                     'Sand',
                     'Script MT',
                     'Felipa',
                     'cursive'],
    'font.family': ['Arial'],
    'font.fantasy': ['Comic Sans MS',
                     'Chicago',
                     'Charcoal',
                     'Impact',
                     'Western',
                     'Humor Sans',
                     'xkcd',
                     'fantasy'],
    'font.monospace': ['DejaVu Sans Mono',
                       'Bitstream Vera Sans Mono',
                       'Computer Modern Typewriter',
                       'Andale Mono',
                       'Nimbus Mono L',
                       'Courier New',
                       'Courier',
                       'Fixed',
                       'Terminal',
                       'monospace'],
    'font.sans-serif': ['DejaVu Sans',
                        'Bitstream Vera Sans',
                        'Computer Modern Sans Serif',
                        'Lucida Grande',
                        'Verdana',
                        'Geneva',
                        'Lucid',
                        'Arial',
                        'Helvetica',
                        'Avant Garde',
                        'sans-serif'],
    'font.serif': ['DejaVu Serif',
                   'Times New Roman',
                   'Bitstream Vera Serif',
                   'Computer Modern Roman',
                   'New Century Schoolbook',
                   'Century Schoolbook L',
                   'Utopia',
                   'ITC Bookman',
                   'Bookman',
                   'Nimbus Roman No9 L',
                   'Times New Roman',
                   'Times',
                   'Palatino',
                   'Charter',
                   'serif'],
    'font.size': 20.0,
    'font.stretch': 'normal',
    'font.style': 'normal',
    'font.variant': 'normal',
    'font.weight': 'bold',
    'grid.alpha': 1.0,
    'grid.color': '#b0b0b0',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'hatch.color': 'black',
    'hatch.linewidth': 1.0,
    'hist.bins': 10,
    'image.aspect': 'equal',
    'image.cmap': 'viridis',
    'image.composite_image': True,
    'image.interpolation': 'nearest',
    'image.lut': 256,
    'image.origin': 'lower',
    'image.resample': True,
    'interactive': True,
    'keymap.all_axes': ['a'],
    'keymap.back': ['left', 'c', 'backspace'],
    'keymap.copy': ['ctrl+c', 'cmd+c'],
    'keymap.forward': ['right', 'v'],
    'keymap.fullscreen': ['f', 'ctrl+f'],
    'keymap.grid': ['g'],
    'keymap.grid_minor': ['G'],
    'keymap.help': ['f1'],
    'keymap.home': ['h', 'r', 'home'],
    'keymap.pan': ['p'],
    'keymap.quit': ['ctrl+w', 'cmd+w', 'q'],
    'keymap.quit_all': ['W', 'cmd+W', 'Q'],
    'keymap.save': ['s', 'ctrl+s'],
    'keymap.xscale': ['k', 'L'],
    'keymap.yscale': ['l'],
    'keymap.zoom': ['o'],
    'legend.borderaxespad': 0.5,
    'legend.borderpad': 0.4,
    'legend.columnspacing': 2.0,
    'legend.edgecolor': '0.8',
    'legend.facecolor': 'inherit',
    'legend.fancybox': True,
    'legend.fontsize': 20.0,
    'legend.framealpha': 0.8,
    'legend.frameon': False,
    'legend.handleheight': 0.7,
    'legend.handlelength': 2.0,
    'legend.handletextpad': 0.8,
    'legend.labelspacing': 0.5,
    'legend.loc': 'best',
    'legend.markerscale': 1.0,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.shadow': False,
    'legend.title_fontsize': 20.0,
    'lines.antialiased': True,
    'lines.color': 'C0',
    'lines.dash_capstyle': 'butt',
    'lines.dash_joinstyle': 'round',
    'lines.dashdot_pattern': [6.4, 1.6, 1.0, 1.6],
    'lines.dashed_pattern': [3.7, 1.6],
    'lines.dotted_pattern': [1.0, 1.65],
    'lines.linestyle': '-',
    'lines.linewidth': 2.0,
    'lines.marker': 'None',
    'lines.markeredgecolor': 'auto',
    'lines.markeredgewidth': 1.0,
    'lines.markerfacecolor': 'auto',
    'lines.markersize': 10.0,
    'lines.scale_dashes': True,
    'lines.solid_capstyle': 'projecting',
    'lines.solid_joinstyle': 'round',
    'markers.fillstyle': 'full',
    'mathtext.bf': 'sans:bold',
    'mathtext.cal': 'cursive',
    'mathtext.default': 'default',
    'mathtext.fallback_to_cm': True,
    'mathtext.fontset': 'custom',
    'mathtext.it': 'sans:italic',
    'mathtext.rm': 'sans',
    'mathtext.sf': 'sans',
    'mathtext.tt': 'monospace',
    'patch.antialiased': True,
    'patch.edgecolor': 'black',
    'patch.facecolor': 'C0',
    'patch.force_edgecolor': False,
    'patch.linewidth': 1.0,
    'path.effects': [],
    'path.simplify': True,
    'path.simplify_threshold': 0.1111111111111111,
    'path.sketch': None,
    'path.snap': True,
    'pdf.compression': 6,
    'pdf.fonttype': 3,
    'pdf.inheritcolor': False,
    'pdf.use14corefonts': False,
    'pgf.preamble': [],
    'pgf.rcfonts': True,
    'pgf.texsystem': 'xelatex',
    'polaraxes.grid': True,
    'ps.distiller.res': 6000,
    'ps.fonttype': 3,
    'ps.papersize': 'letter',
    'ps.useafm': False,
    'ps.usedistiller': False,
    'savefig.bbox': 'tight',
    'savefig.directory': '~',
    'savefig.dpi': 'figure',
    'savefig.edgecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.format': 'png',
    'savefig.jpeg_quality': 300,
    'savefig.orientation': 'portrait',
    'savefig.pad_inches': 0.1,
    'savefig.transparent': True,
    'scatter.marker': 'o',
    'svg.fonttype': 'path',
    'svg.hashsalt': None,
    'svg.image_inline': True,
    'text.antialiased': True,
    'text.color': 'black',
    'text.hinting': 'auto',
    'text.hinting_factor': 8,
    'text.latex.preamble': ['\\boldmath',],
    'text.latex.preview': True,
    'text.usetex': False,
    'timezone': 'UTC',
    'tk.window_focus': False,
    'toolbar': 'toolbar2',
    'xtick.alignment': 'center',
    'xtick.bottom': True,
    'xtick.color': 'black',
    'xtick.direction': 'in',
    'xtick.labelbottom': True,
    'xtick.labelsize': 18.0,
    'xtick.labeltop': False,
    'xtick.major.bottom': True,
    'xtick.major.pad': 3.5,
    'xtick.major.size': 10,
    'xtick.major.top': True,
    'xtick.major.width': 1.5,
    'xtick.minor.bottom': True,
    'xtick.minor.pad': 3.4,
    'xtick.minor.size': 6,
    'xtick.minor.top': True,
    'xtick.minor.visible': True,
    'xtick.minor.width': 1.5,
    'xtick.top': True,
    'ytick.alignment': 'center_baseline',
    'ytick.color': 'black',
    'ytick.direction': 'in',
    'ytick.labelleft': True,
    'ytick.labelright': False,
    'ytick.labelsize': 18.0,
    'ytick.left': True,
    'ytick.major.left': True,
    'ytick.major.pad': 3.5,
    'ytick.major.right': True,
    'ytick.major.size': 10,
    'ytick.major.width': 1.5,
    'ytick.minor.left': True,
    'ytick.minor.pad': 3.4,
    'ytick.minor.right': True,
    'ytick.minor.size': 6.0,
    'ytick.minor.visible': True,
    'ytick.minor.width': 1.5,
    'ytick.right': True}

label = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']

from matplotlib.ticker import FuncFormatter

def my_formatter(x, pos):
    if x==0:
        return x
    else:
        return x
formatter = FuncFormatter(my_formatter)


class MyLocator(matplotlib.ticker.AutoMinorLocator):
    def __init__(self, n=2):
        super().__init__(n=n)

matplotlib.ticker.AutoMinorLocator = MyLocator

def autolabel(ax, fmt="{}", values=None, **kwargs):
    for ind,rect in enumerate(ax.patches):
        x = rect.get_x() + rect.get_width()/2.
        y = rect.get_height()
        if values==None:
            label = y
        else:
            label = values[ind]
        ax.annotate(fmt.format(label), (x,y), xytext=(0,5), textcoords="offset points",
                    ha='center', va='bottom', weight='bold', **kwargs)


def make_custom_legend(patches,texts,*args,**kwargs):
    leg = plt.legend(patches,texts,*args,**kwargs)
    for text,p in zip(leg.get_texts(),patches):
        text.set_color(p.get_fc())


def beautify_leg(leg,color=None):
    # change the font colors to match the line colors:
    if color==None:
        for handle,text in zip(leg.legendHandles, leg.get_texts()):
            try:
                text.set_color(handle.get_color()[0][:3])
            except:
                text.set_color(handle.get_color())
            text.set_text(text.get_text())
    else:
        for handle,text in zip(leg.legendHandles, leg.get_texts()):
            text.set_color(color)
            text.set_text(text.get_text())
    def first_h(h):
        try:
            return h[0]
        except:
            return h
    leg.legendHandles = [first_h(h) for h in leg.legendHandles]
    return leg



    return leg


def panel(rows,cols, figsize=None, size=6, mx=1, my=1, l_p = 0.16, b_p=0.16,r_p = 0.08, t_p=0.08, vs=0.25, hs=0.25, brackets = True,label_align = ['left','top'],
                        hshift=0.75*0.25, vshift = 0.3*0.25,label=label,merge=(),**kwargs):
    global mpl,plt
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(rows, cols)


    labels = label.copy()

    if brackets:
        for ind,i in enumerate(label):
            labels[ind] = __t('({})'.format(i))

    n = cols
    m = rows

    l_p = l_p/n
    r_p = r_p/n
    b_p = b_p/m
    t_p = t_p/m

    fig_y = size*(m+vs*(m-1))/(1-t_p-b_p)
    fig_x = size*(n+hs*(n-1))/(1-l_p-r_p)

    r_p = 1-r_p
    t_p = 1-t_p

    sx = size/fig_x
    sy = size/fig_y

    fig_x *= mx
    fig_y *= my

    if figsize==None:
        figsize=[fig_x,fig_y]
    fig = plt.figure(figsize=figsize, **kwargs)
    fig.set_size_inches(*figsize)
    plt.subplots_adjust(top=t_p, bottom=b_p, left=l_p, right=r_p, hspace=vs,
                        wspace=hs)

    ind = 0
    axis = []
    for i in range(rows):
        for j in range(cols):
            if 0:
                pass
            else:
                ax = plt.subplot(rows,cols,i*cols+j+1)
                if rows*cols==1:
                    pass
                else:
                    ax.text(0-hshift, 1+vshift, '{}'.format(labels[ind]),
                horizontalalignment=label_align[0],verticalalignment=label_align[1],transform=ax.transAxes)

                axis.append(ax)
            ind += 1

    # for ax in axis:
    #     ax.xaxis.set_major_formatter(formatter)
    #     ax.yaxis.set_major_formatter(formatter)

    return fig,axis

def title(ttl,ax = 0,**kwargs):
    global plt
    try:
        iter(ax)
        for i in zip(ttl,ax):
            i[1].set_title(__t(i[0]),**kwargs)
    except:
        if ax==0:
            ax = plt.gca()
        ax.set_title(__t(ttl),**kwargs)

def xlabel(label,ax = 0,**kwargs):
    global plt
    try:
        iter(ax)
        for i in zip(label,ax):
            i[1].set_xlabel(__t(i[0]),**kwargs)
    except:
        if ax==0:
            ax = plt.gca()
        ax.set_xlabel(__t(label),**kwargs)

def ylabel(label,ax = 0,**kwargs):
    global plt
    try:
        iter(ax)
        for i in zip(label,ax):
            i[1].set_ylabel(__t(i[0]),**kwargs)
    except:
        if ax==0:
            ax = plt.gca()
        ax.set_ylabel(__t(label),**kwargs)


def xticks(pos,ax = None,labels = None,**kwargs):
    global plt,font
    if ax==None:
        ax = plt.gca()
    ax.set_xticks(pos,**kwargs)
    if labels==None:
        pass
    else:
        ax.set_xticklabels(['{}'.format(i) for i in labels])

def yticks(pos,ax = None,labels = None,**kwargs):
    global plt,font
    if ax==None:
        ax = plt.gca()
    ax.set_yticks(pos,**kwargs)
    if labels==None:
        pass
    else:
        ax.set_yticklabels(['{}'.format(i) for i in labels])

def xlim(lim,ax = None,**kwargs):
    global plt
    if ax==None:
        ax = plt.gca()
    ax.set_xlim(lim,**kwargs)

def ylim(lim,ax = None,**kwargs):
    global plt
    if ax==None:
        ax = plt.gca()
    ax.set_ylim(lim,**kwargs)

def vertical(x0,line='',ax = None,**kwargs):
    global plt
    if ax==None:
        ax = plt.gca()
    ax.axvline(x0,**kwargs)

def horizontal(y0,line='',ax = None,**kwargs):
    global plt
    if ax==None:
        ax = plt.gca()
    ax.axhline(y0,**kwargs)


def legend_off(ax=None):
    global plt
    if ax==None:
        ax = plt.gca()
    ax.get_legend().remove()

def legend_on(ax=None,color=None,**kwargs):
    global plt
    if ax==None:
        ax = plt.gca()
    leg = ax.legend(frameon=False,**kwargs) #loc=1,bbox_to_anchor=[0.95,0.95],)
    return beautify_leg(leg,color)

def legend_on2(ax=None,color=None,frameon=False,**kwargs):
    global plt
    if ax==None:
        ax = plt.gca()
    # get handles
    handles, labels = ax.get_legend_handles_labels()
    # remove the errorbars
    def first_h(h):
        try:
            return h[0]
        except:
            return h
    handles = [first_h(h) for h in handles]
    # use them in the legend
    leg = ax.legend(handles, labels, frameon=frameon,**kwargs) #loc=1,bbox_to_anchor=[0.95,0.95],)
    return beautify_leg(leg,color)


def sticky_legend(pos,c=0,ax=None):
    global plt
    if ax==None:
        ax = plt.gca()
    legend_on(ax)
    lines = ax.get_legend().get_lines()
    texts = ax.get_legend().get_texts()
    ax.annotate('{}'.format(texts[c].get_text()), xy=(pos[0], pos[1]), color=lines[c].get_color())
    legend_off(ax)

def twinx(ax,color=['b','r']):
    global plt
    ax2 = ax.twinx()
    ax.spines['left'].set_edgecolor(color[0])
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y',colors=color[0],which='both')
    ax2.spines['right'].set_edgecolor(color[1])
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='y',colors=color[1],which='both')
    return ax2,ax

def twiny(ax,color=['b','r']):
    global plt
    ax2 = ax.twiny()
    ax.spines['bottom'].set_edgecolor(color[0])
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x',colors=color[0],which='both')
    ax2.spines['top'].set_edgecolor(color[1])
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x',colors=color[1],which='both')
    ax2.set_yticklabels(horizontalalignment='right')
    return ax2

def zero_axis(color='k',alpha=0.2,linewidth = 0.5,**kwargs):
    horizontal(y0=0,color=color,alpha=alpha,linewidth=linewidth,**kwargs)
    vertical(x0=0,color=color,alpha=alpha,linewidth=linewidth,**kwargs)

def arrow(ax=None,pos=[[0,0],[1,1]],fc='k',alpha=1,curve=0,**kwargs):
    global plt
    if ax==None:
            ax = plt.gca()
    connectionstyle="arc3,rad={}".format(curve)
    arrowprops=dict(fc = fc, alpha = alpha,**kwargs,connectionstyle=connectionstyle)
    ax.annotate("", xytext=pos[0], xy=pos[1],arrowprops=arrowprops)

def annot(text,point,put,ax=None,connectionstyle=None,fc='k',alpha=1,curve=0,**kwargs):
    global plt
    if ax==None:
            ax = plt.gca()
    if connectionstyle==None:
        connectionstyle="arc3,rad={}".format(curve)
    arrowprops=dict(fc = fc, alpha = alpha,**kwargs,connectionstyle=connectionstyle)
    ax.annotate('{}'.format(text), xytext=put, xy=point,arrowprops=arrowprops,**kwargs)

def text(x,y,text,ax=None,**kwargs):
    global plt
    if ax==None:
            ax = plt.gca()
    return ax.text(x,y,__t(text), **kwargs)

def savefig(filename,dpi=80,**kwargs):
    global plt
    plt.savefig(filename,dpi=dpi,**kwargs)

def show():
    global plt
    plt.show()

def padding(l_p = 0.1, r_p = 0.05, b_p=0.1, t_p=0.05, vs=0.1, hs=0, size=6, **kwargs):
    n = 1
    m = 1
    r_p = 1-r_p/n
    t_p = 1-t_p/m
    fig_y = m*size/(t_p-b_p-(m-1)*vs)
    fig_y = m*size/(t_p-b_p-(m-1)*vs*size/fig_y)
    fig_y = m*size/(t_p-b_p-(m-1)*vs*size/fig_y)
    fig_x = n*size/(r_p-l_p-(n-1)*hs)
    fig_x = n*size/(r_p-l_p-(n-1)*hs*size/fig_x)
    fig_x = n*size/(r_p-l_p-(n-1)*hs*size/fig_x)

    global plt
    fig = plt.gcf()
    fig.set_size_inches(fig_x,fig_y)
    plt.subplots_adjust(top=t_p, bottom=b_p, left=l_p, right=r_p, hspace=vs,
                            wspace=hs,**kwargs)

def bar_plot(data,ax=None,x_ind=0,label=None,rot=0,fontsize=14,fmt="{}",
                text=False, hatch=None,
                fig_size=6,l_p = 0.16, b_p=0.16,r_p = 0.05, t_p=0.05, vs=0.25, hs=0.25,
                                        hshift=0.75*0.25, vshift = 0.3*0.25, **kwargs):
    global np, plt, mpl
    data = numpy.array(data)
    if data.ndim==1:
        data = [data]
    if hatch==None:
        hatch = ['']*len(data)

    if plt.get_fignums():
        fig = plt.gcf()
    else:
        fig,ax = panel(1,1,size=fig_size,l_p = l_p, r_p = r_p, b_p=b_p, t_p=t_p, vs=vs, hs=hs, )
    ax = plt.gca()

    width = (numpy.diff(data[x_ind])).min()
    if width<=0:
        print('x values are not consistant.')
    else:
        len1 = len(data)-1
        width=width/(len1+1)
        incr = 0
        for ind,y in enumerate(data):
            if ind==x_ind:
                pass
            else:
                x = numpy.array(data[x_ind])-len1*width/2+width/2+width*incr
                y = [i for i in data[ind]]
                ax.bar(x,y,width=width,label=label[ind],hatch = hatch[ind],**kwargs)
                ax.tick_params(top=False,bottom=False,which='both')
                if text:
                    autolabel(ax,rotation=rot,size=fontsize,fmt=fmt)
                incr += 1

def line_plot(x,y,*args,fig_size=6,l_p = 0.16, b_p=0.16,r_p = 0.05, t_p=0.05, vs=0.25, hs=0.25,
                        hshift=0.75*0.25, vshift = 0.3*0.25, **kwargs):
    global mpl,plt

    if plt.get_fignums():
        fig = plt.gcf()
    else:
        fig,ax = panel(1,1,size=fig_size,l_p = l_p, r_p = r_p, b_p=b_p, t_p=t_p, vs=vs, hs=hs, )
    ax = plt.gca()
    ax.plot(x,y,*args,**kwargs)

    return fig, ax

def put_image(image_file,*args,fig_size=6,l_p = 0.16, b_p=0.16,r_p = 0.05, t_p=0.05, vs=0.25, hs=0.25,
                            hshift=0.75*0.25, vshift = 0.3*0.25, **kwargs):
        global mpl,plt

        if plt.get_fignums():
            fig = plt.gcf()
        else:
            fig,ax = panel(1,1,size=fig_size,l_p = l_p, r_p = r_p, b_p=b_p, t_p=t_p, vs=vs, hs=hs, )
        ax = plt.gca()
        image = plt.imread(image_file)
        ax.imshow(image[::-1],*args,**kwargs)
        ax.set_axis_off()

        return fig, ax

def h_strip(y0,y1,ax=None,color='k',alpha=0.2,**kwargs):
    global plt
    if ax==None:
        ax = plt.gca()
    ax.axhspan(y0,y1,color=color,alpha=alpha,**kwargs)

def v_strip(x0,x1,ax=None,color='k',alpha=0.2,**kwargs):
    global plt
    if ax==None:
        ax = plt.gca()
    ax.axvspan(x0,x1,color=color,alpha=alpha,**kwargs)


def makePatch(vertices,ax=None,fc='grey',ec='none',alpha=0.2,curve=0,**kwargs):
    global plt
    if ax==None:
        ax = plt.gca()
    incr = {0:Path.LINETO,1:Path.CURVE3,2:Path.CURVE4}
    codes = []
    vertices_all = []
    for vert in vertices:
        codes = [Path.MOVETO] + [incr[curve]]*(len(vert)-1) + [Path.CLOSEPOLY]
        vertices_all += list(vert) + [vert[0]]

    vertices_all = numpy.array(vertices_all, float)
    path = Path(vertices_all, codes)

    pathpatch = PathPatch(path, facecolor=fc, edgecolor=ec,alpha=alpha,**kwargs)
    ax.add_patch(pathpatch)
    return pathpatch

def rectangle(xlo,xhi,ylo,yhi,ax=None,**kwargs):
    global plt
    if ax==None:
            ax = plt.gca()
    vertices = [[xlo,ylo],[xlo,yhi],[xhi,yhi],[xhi,ylo]]
    return makePatch([vertices],ax=ax,**kwargs)

def polygon(origin,radius,sides=3,y_scale=0,b=1,b_=1,rot=0,ax=None,**kwargs):
    global plt
    if ax==None:
            ax = plt.gca()
    vertices = []
    range_x = ax.get_xlim()
    range_y = ax.get_ylim()
    if y_scale==0:
        y_scale = numpy.abs((range_y[1]-range_y[0])/(range_x[1]-range_x[0]))
    else:
        b = 1
        b_ = 1

    theta = rot/180*numpy.pi
    for i in numpy.arange(0,1,1/sides)*2*numpy.pi:
        x = radius*numpy.cos(i)
        y = b*radius*numpy.sin(i)
        vertices.append([origin[0]+x*numpy.cos(theta)+y*numpy.sin(theta), origin[1]+y_scale*b_*(-x*numpy.sin(theta)+y*numpy.cos(theta))])
    return makePatch([vertices],ax=ax,**kwargs)

def ellipse(origin,radius,b = 1, **kwargs):
    return polygon(origin,radius,b = b, sides = 100,**kwargs)

def linestyles(color = ['blue','red','green','purple','brown','magenta','black'], ls = ['-'], ms = ['']):
    global plt,cycler
    if len(ls)==1:
        ls = ls * len(color)
    if len(ms)==1:
        ms = ms * len(color)

    plt.rc('axes', prop_cycle=(cycler('color', color) + cycler('linestyle', ls) + cycler('marker', ms) ) )
    return (cycler('color', color) + cycler('linestyle', ls) + cycler('marker', ms) )


def grid_on():
    global plt
    plt.grid()

def grid_off():
    global plt
    plt.grid(False)

def linear(x,m,c):
    return m*numpy.array(x) + c

def fit(x_data,y_data,xs,*args,func=linear,precision=2,spline=False,skip_fit=False, label=r'${}x+{}$',**kwargs):
    from scipy.optimize import curve_fit

    if skip_fit:
        ys = y_data
    else:
        popt, pcov = curve_fit(func,x_data,y_data)
        ys = func(numpy.array(xs),*popt)
        params = [ '{:.{}f}'.format(p[0],p[1]) for p in zip(popt,precision)]
        label = label.format(*params)
        try:
            len(precision)
        except:
            precision = [precision]*len(popt)

    if spline:
        tck,u = interpolate.splprep([xs,ys],s=0)
        xs,ys1 = interpolate.splev(numpy.linspace(0,1,100),tck,der=0)
    else:
        ys1 = ys

    line_plot(xs,ys1,*args,label=label,**kwargs)
    return ys,label

def set_markersize(ms):
    for i in plt.rcParams:
        if 'markersize' in i:
            plt.rcParams[i] = ms

def inset(bounds, labelsize=16, ax = None, **kwargs):

    global plt,mpl

    if ax == None:
        ax = plt.gca()
    in_ax = ax.inset_axes(bounds=bounds)
    ax.figure.add_axes(in_ax)

    in_ax.tick_params(axis= 'both',labelsize=labelsize)

    return in_ax

def zoomed_in(x,y,*args,bounds=[0.15,0.6,0.35,0.35],labelsize=16, ax = None, connect = True, loc1=2, loc2=4,loc3=None,loc4=None,prop={},**kwargs):
    prop1 = dict(fc="grey", ec="grey",alpha=0.2)
    prop1.update(prop)

    global plt,mpl
    if loc3 == None:
        loc3 = loc1
    if loc4 == None:
        loc4 = loc2

    if ax == None:
        ax = plt.gca()
    in_ax = ax.inset_axes(bounds=bounds)
    ax.figure.add_axes(in_ax)



    in_ax.tick_params(axis= 'both',labelsize=labelsize)

    in_ax.plot(x,y,*args,**kwargs)
    aspect_ratio = numpy.diff(ax.get_ylim())/numpy.diff(ax.get_xlim())
    aspect_ratio2 = numpy.diff(in_ax.get_ylim())/numpy.diff(in_ax.get_xlim())

    if aspect_ratio>aspect_ratio2:
        Ylim = in_ax.get_ylim()
        in_ax.set_ylim([numpy.mean(Ylim)-aspect_ratio/aspect_ratio2*numpy.diff(Ylim)/2, numpy.mean(Ylim)+aspect_ratio/aspect_ratio2*numpy.diff(Ylim)/2,])

    if aspect_ratio2>aspect_ratio:
        Xlim = in_ax.get_xlim()
        in_ax.set_xlim([numpy.mean(Xlim)-aspect_ratio2/aspect_ratio*numpy.diff(Xlim)/2, numpy.mean(Xlim)+aspect_ratio2/aspect_ratio*numpy.diff(Xlim)/2,])

    if connect:
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        mark_inset(ax, in_ax, loc1=loc1, loc2=loc2,**prop1,)
        mark_inset(ax, in_ax, loc1=loc3, loc2=loc4,**prop1,)

    return in_ax


def set_things():
    linestyles()
    mpl.rcParams.update(Params_)



import math
import matplotlib.pyplot as plt
class AnnoteFinder(object):
    """callback for matplotlib to display an annotation when points are
    clicked on.  The point which is closest to the click and within
    xtol and ytol is identified.

    Register this function like this:

    scatter(xdata, ydata)
    af = AnnoteFinder(xdata, ydata, annotes)
    connect('button_press_event', af)

    example:

    x = range(10)
    y = range(10)
    annotes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    fig, ax = plt.subplots()
    ax.scatter(x,y)
    af =  AnnoteFinder(x,y, annotes, ax=ax)
    fig.canvas.mpl_connect('button_press_event', af)
    plt.show()
    """

    def __init__(self, xdata, ydata, annotes, ax=None, xtol=None, ytol=None):
        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None:
            xtol = ((max(xdata) - min(xdata))/float(len(xdata)))/2
        if ytol is None:
            ytol = ((max(ydata) - min(ydata))/float(len(ydata)))/2
        self.xtol = xtol
        self.ytol = ytol
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.drawnAnnotations = {}
        self.links = []

    def distance(self, x1, x2, y1, y2):
        """
        return the distance between two points
        """
        return(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))

    def __call__(self, event):

        if event.inaxes:

            clickX = event.xdata
            clickY = event.ydata
            if (self.ax is None) or (self.ax is event.inaxes):
                annotes = []
                # print(event.xdata, event.ydata)
                for x, y, a in self.data:
                    # print(x, y, a)
                    if ((clickX-self.xtol < x < clickX+self.xtol) and
                            (clickY-self.ytol < y < clickY+self.ytol)):
                        annotes.append(
                            (self.distance(x, clickX, y, clickY), x, y, a))
                if annotes:
                    annotes.sort()
                    distance, x, y, annote = annotes[0]
                    self.drawAnnote(event.inaxes, x, y, annote)
                    for l in self.links:
                        l.drawSpecificAnnote(annote)

    def drawAnnote(self, ax, x, y, annote):
        """
        Draw the annotation on the plot
        """
        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.ax.figure.canvas.draw_idle()
        else:
            t = ax.text(x, y, " - %s" % (annote),)
            m = ax.scatter([x], [y], marker='d', c='r', zorder=100)
            self.drawnAnnotations[(x, y)] = (t, m)
            self.ax.figure.canvas.draw_idle()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self.data if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.ax, x, y, a)

def linkAnnotationFinders(afs):
    r'''

    subplot(121)
    scatter(x,y)
    af1 = AnnoteFinder(x,y, annotes)
    connect('button_press_event', af1)

    subplot(122)
    scatter(x,y)
    af2 = AnnoteFinder(x,y, annotes)
    connect('button_press_event', af2)

    linkAnnotationFinders([af1, af2])
    '''
    for i in range(len(afs)):
        allButSelfAfs = afs[:i]+afs[i+1:]
        afs[i].links.extend(allButSelfAfs)

def colorbar(cb,ax=None,where="right",size="5%",pad=0.1,**kwargs):
    if ax==None:
        ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(where, size=size, pad=pad, **kwargs)
    return plt.colorbar(cb, cax=cax,)


def smooth_curve(x,y,**kwargs):
    import numpy as np
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    tck,u = interpolate.splprep([x,y], **kwargs)
    unew = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(unew, tck)
    return out[0], out[1]

def spline_plot(x,y,*args,sprop=None, **kwargs):
    if sprop==None:
        sprop = {'k':1,'s':0}
    x, y = smooth_curve(x,y,**sprop)
    line_plot(x,y,*args,**kwargs)


def scatter2patch(x,y,*arg,ax=None,ms=0.01,sides=3,scalex=1,unary=True,**kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.patches as ptc
    from shapely.affinity import scale
    from shapely.geometry import Point
    from shapely.ops import cascaded_union, unary_union
    polygons = [Point(x[i], y[i]).buffer(ms,sides) for i in range(len(x))]
    polygons = [scale(p,xfact=scalex,yfact=1) for p in polygons]
    if unary:
        polygons = unary_union(polygons)
    else:
        polygons = cascaded_union(polygons)

    if ax==None:
        ax = plt.gca()
    try:
        for polygon in polygons:
            polygon = ptc.Polygon(numpy.array(polygon.exterior), **kwargs)
            ax.add_patch(polygon)
    except:
        polygon = ptc.Polygon(numpy.array(polygons.exterior), **kwargs)
        ax.add_patch(polygon)

def paper(*args,**kwargs):
    linestyles(color=['k'])
    fig, [ax] = panel(1,1,r_p=0,l_p=0,t_p=0,b_p=0,**kwargs)
    ax.axis('off')
    return fig, ax


plt.rcParams.update(Params_)


def book_format():
    set_mood_light()
    set_markersize(5)
    set_font(weight='normal')

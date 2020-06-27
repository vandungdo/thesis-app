import pandas as pd
import os
import numpy as np

from bokeh.io import show,curdoc
from bokeh.plotting import figure
import bokeh.plotting as bpl
from bokeh.embed import components
import bokeh.models as bmo
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.palettes import *
import datetime
from math import pi
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    Legend
)

currentPath = os.path.abspath(os.path.dirname(__file__))

df_svm = pd.read_csv(os.path.join(currentPath,'data/percentage_conf_df_svm_balanced.csv'))
df_decision_tree = pd.read_csv(os.path.join(currentPath,'data/percentage_conf_df_decision_trees_balanced.csv'))
df_naive_bayes = pd.read_csv(os.path.join(currentPath,'data/percentage_conf_df_gaussian_naive_bayes_balanced.csv'))
df_random_forest = pd.read_csv(os.path.join(currentPath,'data/percentage_conf_df_random_forest_balanced.csv'))


def heatmap(algorithm):
    if algorithm == 'SVM':
        df = df_svm.copy()
    elif algorithm == 'Decision Tree':
        df = df_decision_tree.copy()
    elif algorithm == 'Naive Bayes':
        df = df_naive_bayes.copy()
    else:
        df = df_random_forest.copy()
    df = df.set_index('Unnamed: 0')
    df.loc[:,'actual'] = list(df.index)
    df['actual'] = df['actual'].astype(str)
    df = df.set_index('actual')
    df.columns.name = 'predict'
    actual = list(df.columns)
    predict = list(df.index)
    df_new = pd.DataFrame(df.stack(), columns=['number']).reset_index() 
    colors = list(reversed(BuPu6))
    mapper = LinearColorMapper(palette=colors, low=df_new.number.min(), high=df_new.number.max())

    source = ColumnDataSource(df_new)

    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    # p is a bokeh figure object
    p = figure(title="Confusion matrix in percentage of "+algorithm,
               x_range=predict, y_range=list(reversed(actual)),
               x_axis_location="above", plot_width=1000, plot_height=600,
               tools=TOOLS, toolbar_location='below',tooltips=[('percentage', '@number')])

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3
    p.xaxis.axis_label = 'Predicted Classes'
    p.yaxis.axis_label = 'Actual Classes'
    p.rect(x="predict", y="actual", width=1, height=1,
           source=source,
           fill_color={'field': 'number', 'transform': mapper},
           line_color=None)

    curdoc().theme = 'dark_minimal'
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="10pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    script, div = components(p)
    return script, div


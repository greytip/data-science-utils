from bokeh.charts import HeatMap, Bar, BoxPlot, Histogram, Scatter, Line, Bar
from bokeh.plotting import figure, show, output_file, output_notebook, ColumnDataSource
from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.models import ( Text, PanTool, WheelZoomTool, LinearAxis,
                           SingleIntervalTicker, Range1d,  Plot,
                           Text, Circle, HoverTool, Triangle)
from math import ceil

BOKEH_TOOLS = "resize,crosshair,pan,wheel_zoom,box_zoom,reset,tap,previewsave,box_select,poly_select,lasso_select"

def timestamp(datetimeObj):
    timestamp = (datetimeObj - datetime(1970, 1, 1)).total_seconds()
    return timestamp


def month_year_format(datetimeObj):
    return str(datetimeObj.strftime("%b-%Y"))


# Normalize values to 0 to 100
#transCountsNorm = transCounts/np.max(np.abs(allCounts), axis=0)
#accountCountsNorm = accountCounts/np.max(np.abs(allCounts), axis=0)

def plot_bar_from_query(conn, query, condition=0, title="Bar chart", xlabel="x", ylabel="y"):
    """
    :param conn: Sqlalchemy connection object
    :param query: query as a string for which you want to plot bar and it must return x and y columns
    :param condition: It is a number. x values in the graph will be greater than condition value.
    :param title: Title of the chart
    :param xlabel: x label for the chart
    :param ylabel: y label for the chart
    :return: script and div (script,div)
    """
    result = conn.execute(query)
    plot_data = {}
    y, cities = [], []
    for row in result:
        if row[0] > condition and row[1]:
            y.append(float(row[0]))
            cities.append(str(row[1]))

    plot_data['y'] = y
    plot = Bar(plot_data, cities,
               title=title,
               xlabel=xlabel, ylabel=ylabel,
               width=470, height=500)
    script, div = components(plot, CDN)
    return script, div


def plot_twin_y_axis_scatter(conn, query1=None, query2=None,
                             xy1={}, xy2={}):
    """
    Plots twin y axis scatter plot you just have to give conn sqlalchemy obj and
    two query/dictionary of x and y values
    :param conn: Sqlaclhemy connection object
    :param query1: query 1 for x and y1
    :param query2: query 2 for x and y2
    :param xy1: dictionary containing x and y key values
    :param xy2: dictionary containing x and y key values
    :return: Bokeh plot object (script,div)
    """

    if query1:
        result = conn.execute(query1)
        plot_data1 = {'x': [], 'y': []}
        for row in result:
            if row[0] and row[1]:
                plot_data1['x'].append(float(row[0]))
                plot_data1['y'].append(str(row[1]))
    else:
        if isinstance(xy1, dict) and xy1:
            plot_data1 = xy1
        else:
            raise ValueError('Parameters values not given properly')

    if query2:
        result = conn.execute(query2)
        plot_data2 = {'x': [], 'y': []}
        for row in result:
            if row[0] and row[1]:
                plot_data2['x'].append(float(row[0]))
                plot_data2['y'].append(str(row[1]))
    else:
        if isinstance(xy2, dict) and xy2:
            plot_data2 = xy2
        else:
            raise ValueError('Parameters values not given properly')

    renderer_source = ColumnDataSource({'x': plot_data1['x'], 'y': plot_data1['y']})
    renderer_source2 = ColumnDataSource({'x': plot_data2['x'], 'y': plot_data2['y']})

    bokeh_plot = BokehTwinLinePlot(plot_data1, plot_data2,
                                   xlabel="No. of Accounts",
                                   ylabel="No. of. Leave Transactions",
                                   ylabel2="No. of services")
    plot = bokeh_plot.get_plot()
    plot = bokeh_plot.add_text(plot)
    # Add the triangle
    triangle_glyph = Triangle(
        x='x', y='y', size=15,
        fill_color='#4682B4', fill_alpha=0.8,
        line_color='#4682B4', line_width=0.5, line_alpha=0.5)
    # Add the circle
    circle_glyph = Circle(
        x='x', y='y', size=15,
        fill_color='#d24726', fill_alpha=0.8,
        line_color='#d24726', line_width=0.5, line_alpha=0.5)
    triangle_renderer = plot.add_glyph(renderer_source, triangle_glyph)
    circle_renderer = plot.add_glyph(renderer_source2, circle_glyph, y_range_name="y_range2")

    # Add the hover (only against the circle and not other plot elements)
    tooltips = "@index"
    plot.add_tools(HoverTool(tooltips=tooltips, renderers=[triangle_renderer, circle_renderer]))
    plot.add_tools(PanTool(), WheelZoomTool())
    return plot


def roundup(x):
    """
    :param x:
    :return: round up the value
    """
    return int(ceil(x / 10.0))*2


class BokehTwinLinePlot(object):
    """
    Class for creating basic bokeh structure of two y axis and one x axis
    """

    def __init__(self, plot_data1, plot_data2, xlabel='x', ylabel='y', ylabel2='y2'):
        self.plot_data1 = plot_data1
        self.plot_data2 = plot_data2
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.ylabel2 = ylabel2

    def get_plot(self):
        """
        Creates the basic bokeh plot with xrange, y range
        :return: Boekh plot obj.
        """
        min_x_range, max_x_range = self.get_x_ranges()
        min_y_range, max_y_range = self.get_y_ranges(self.plot_data1)
        min_y2_range, max_y2_range = self.get_y_ranges(self.plot_data2)
        xdr = Range1d(min_x_range-(min_x_range/1.2), max_x_range+(max_x_range/1.2))
        ydr = Range1d(min_y_range-(min_y_range/1.2), max_y_range+(max_y_range/1.2))
        ydr2 = Range1d(min_y2_range-(min_y2_range/10), max_y2_range+(max_y2_range/10))
        plot = Plot(
            x_range=xdr,
            y_range=ydr,
            extra_y_ranges={"y_range2":ydr2},
            title="",
            plot_width=550,
            plot_height=550,
            outline_line_color=None,
            toolbar_location=None,
        )
        return plot

    def add_axes(self, plot):
        """
        Adds axis to Bokeh plot Obj
        :param plot: Bokeh plot obj. from get_plot method
        :return: Bokeh plot obj
        """
        min_x_range, max_x_range = self.get_x_ranges()
        min_y_range, max_y_range = self.get_y_ranges(self.plot_data1)
        min_y2_range, max_y2_range = self.get_y_ranges(self.plot_data2)
        x_interval = roundup(max_x_range)
        y_interval = roundup(max_y_range)
        y2_interval = roundup(max_y2_range)
        xaxis = LinearAxis(SingleIntervalTicker(interval=x_interval), axis_label=self.xlabel, **AXIS_FORMATS)
        yaxis = LinearAxis(SingleIntervalTicker(interval=y_interval), axis_label=self.ylabel, **AXIS_FORMATS)
        yaxis2 = LinearAxis(SingleIntervalTicker(interval=y2_interval), y_range_name="y_range2", axis_label=self.ylabel2, **AXIS_FORMATS)
        plot.add_layout(xaxis, 'below')
        plot.add_layout(yaxis, 'left')
        plot.add_layout(yaxis2, 'right')
        return plot

    def add_text(self, plot):
        """
        Adds text to Bokeh plot
        :param plot: Bokeh plot obj.
        :return: Bokeh plot obj.
        """
        plot = self.add_axes(plot)
        return plot

    def get_x_ranges(self):
        """
        get the minimum and maximum values of x
        :return: Minimum x value, Maximum x value
        """
        plot_data1_x = list(self.plot_data1['x'])
        plot_data2_x = list(self.plot_data2['x'])
        if not plot_data1_x:
            plot_data1_x = [0]
        if not plot_data2_x:
            plot_data2_x = [0]
        min_x_range = min([min(plot_data1_x),min(plot_data2_x)])
        max_x_range = max([max(plot_data1_x), max(plot_data2_x)])

        return min_x_range, max_x_range

    def get_y_ranges(self, plot_data):
        """
        get the minimum and maximum values of y
        :return: Minimum y value, Maximum y value
        """
        plot_data_y = map(float, list(plot_data['y']))
        if not plot_data_y:
            plot_data_y = [0]
        min_y_range = min(plot_data_y)
        max_y_range = max(plot_data_y)
        return min_y_range, max_y_range

# Axis settings for Bokeh plots
AXIS_FORMATS = dict(
    minor_tick_in=None,
    minor_tick_out=None,
    major_tick_in=None,
    major_label_text_font_size="10pt",
    major_label_text_font_style="normal",
    axis_label_text_font_size="10pt",

    axis_line_color='#AAAAAA',
    major_tick_line_color='#AAAAAA',
    major_label_text_color='#666666',

    major_tick_line_cap="round",
    axis_line_cap="round",
    axis_line_width=1,
    major_tick_line_width=1,)


def c3plot_from_query(conn, query, condition=0):
    """
    :param conn: Sqlalchemy connection object
    :param query: query as a string for which you want to plot bar and it must return x and y columns
    :param condition: It is a number. x values in the graph will be greater than condition value.
    :return: script and div (script,div)
    """
    result = conn.execute(query)
    plot_data = {}
    y, cities = [], []
    for row in result:
        if row[0] > condition and row[1]:
            y.append(float(row[0]))
            cities.append(str(row[1]))

    plot_data['y'] = y
    return cities, plot_data['y']
def histogram(histDF,values, **kwargs):
    return Histogram(histDF[values], **kwargs)

def barplot(barDF, xlabel, ylabel, title="Bar Plot",
                            agg='sum', **kwargs):
    barplot = Bar(barDF, xlabel, values=ylabel, agg=agg, title=title, **kwargs)
    return barplot

def boxplot(boxDF, values_label, xlabel, title="boxplot", **kwargs):
    boxplot = BoxPlot(boxDF, values=values_label, label=xlabel, color=xlabel, title=title, **kwargs)
    return boxplot

def heatmap(heatMapDF,xlabel, ylabel, value_label,title="heatmap", palette=None, **kwargs):
    if not palette:
        from bokeh.palettes import RdBu11 as palette_tmpl
        palette = palette_tmpl
    hm = HeatMap(heatMapDF, x=xlabel, y=ylabel, values=value_label,
                        title=title, width=800, palette=palette, **kwargs)
    return hm

def scatterplot(scatterDF, xcol, ycol, xlabel, ylabel, group=None, **kwargs):
    if not group:
        scatter = Scatter(scatterDF, x=xcol, y=ycol, xlabel=xlabel, ylabel=ylabel, **kwargs)
    else:
        scatter = Scatter(scatterDF, x=xcol, y=ycol, xlabel=xlabel,
                                ylabel=ylabel, color=group, **kwargs)
    return scatter

def mscatter(p, x, y, typestr):
    p.scatter(x, y, marker=typestr,
            line_color="#6666ee", fill_color="#ee6666", fill_alpha=0.5, size=12)


def mtext(p, x, y, textstr):
    p.text(x, y, text=[textstr],
         text_color="#449944", text_align="center", text_font_size="10pt")

def boxplot(xrange, yrange, boxSource, xlabel='x', ylabel='y', colors=list()):
    p=figure(
        title='\"Party\" Disturbance Calls in LA',
        x_range=xrange,
        y_range=yrange)
        #tools=TOOLS)

    p.plot_width=900
    p.plot_height = 400
    p.toolbar_location='left'

    p.rect(xlabel, ylabel, 1, 1, source=boxSource, color=colors, line_color=None)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi/3

    #hover = p.select(dict(type=HoverTool))
    #hover.tooltips = OrderedDict([
    #    ('parties', '<a href='https://github.com/parties' class='user-mention'>@parties</a>'),
    #])
    return p

def sb_boxplot(dataframe, quant_field, cat_fields=None, facetCol=None ):
    assert quant_field
    if cat_fields:
        assert(len(cat_fields) <=2)
        #assert(all([isinstance(dataframe[field].dtype, str) for  field in cat_fields]))
    import seaborn as sns
    sns.set_style("whitegrid")
    tips = sns.load_dataset("tips")
    if not facetCol:
        if not cat_fields:
            ax = sns.boxplot(dataframe[quant_field])
        elif len(cat_fields)==1:
            sns.boxplot(x=cat_fields[0], y=quant_field, data=dataframe)
        else:
            sns.boxplot(x=cat_fields[0], y=quant_field,
                        hue=cat_fields[1], data=dataframe,
                        palette="Set3", linewidth=2.5)
    else:
        fg = sns.FacetGrid(dataframe, col=facetCol, size=4, aspect=7)
        (fg.map(sns.boxplot, cat_fields[0], quant_field,cat_fields[1])\
               .despine(left=True)\
               .add_legend(title=cat_fields[1]))

def sb_heatmap(df, label):
    # Creating heatmaps in matplotlib is more difficult than it should be.
    # Thankfully, Seaborn makes them easy for us.
    # http://stanford.edu/~mwaskom/software/seaborn/

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style='white')

    plt.figure(figsize=(12, 8))
    plt.title(label)
    sns.heatmap(df.T, mask=df.T.isnull(), annot=True, fmt='.0%');

def sb_violinplot(df,column):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Compute the correlation matrix and average over networks
    corr_df = df.corr().groupby(level=column).mean()
    corr_df.index = corr_df.index.astype(int)
    corr_df = corr_df.sort_index().T

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 6))

    # Draw a violinplot with a narrower bandwidth than the default
    sns.violinplot(data=corr_df, palette="Set3", bw=.2, cut=1, linewidth=1)

    # Finalize the figure
    ax.set(ylim=(-.7, 1.05))
    sns.despine(left=True, bottom=True)

def sb_jointplot(series1, series2):
    import numpy as np
    import seaborn as sns
    sns.set(style="white")

    # Generate a random correlated bivariate dataset
    #rs = np.random.RandomState(5)
    #mean = [0, 0]
    #cov = [(1, .5), (.5, 1)]
    #x1, x2 = rs.multivariate_normal(mean, cov, 500).T
    #x1 = pd.Series(x1, name="$X_1$")
    #x2 = pd.Series(x2, name="$X_2$")

    # Show the joint distribution using kernel density estimation
    return sns.jointplot(series1, series2, kind="kde", size=7, space=0)

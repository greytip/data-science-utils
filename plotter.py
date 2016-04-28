from bokeh.charts import HeatMap, Bar, BoxPlot, Histogram, Scatter
from bokeh.plotting import figure, show, output_file, output_notebook, ColumnDataSource

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




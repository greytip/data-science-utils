# Type checkers taken from here. http://stackoverflow.com/questions/25039626/find-numeric-columns-in-pandas-python
def is_type(df, baseType):
    import numpy as np
    import pandas as pd
    test = [issubclass(np.dtype(d).type, baseType) for d in df.dtypes]
    return pd.DataFrame(data = test, index = df.columns, columns = ["test"])

def calculate_anova(df, targetCol, sourceCol):
    from statsmodels.formula.api import ols
    lm = ols('conformity ~ C(%s, Sum)*C(%s, Sum)'% (targetCol, sourceCol),
            data=df).fit()
    table = sm.stats.anova_lm(lm, typ=2)
    return table

def is_float(df):
    import numpy as np
    return is_type(df, np.float)

def is_number(df):
    import numpy as np
    return is_type(df, np.number)

def is_integer(df):
    import numpy as np
    return is_type(df, np.integer)

def apply_on_all(seq, method, *args, **kwargs):
    """
    Simply apply a method on all objects in the sequence.
    Based on: http://stackoverflow.com/questions/2682012/how-to-call-same-method-for-a-list-of-objects
    """
    result = list()
    for obj in seq:
        result.append(getattr(obj, method)(*args, **kwargs))
    return result

def chunks(combos, size=9):
    for i in range(0, len(combos), size):
        yield combos[i:i + size]

def get_figures_and_combos(combos):
    from matplotlib import pyplot as plt
    figures = list()
    combo_lists = list()
    for each in chunks(combos, 9):
        figures.append(plt.figure(figsize=(20,10)))
        combo_lists.append(each)
    return figures, combo_lists

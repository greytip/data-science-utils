
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

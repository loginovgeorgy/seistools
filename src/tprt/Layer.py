def _iso_model(vp=3500, vs=3500, *args, **kwargs):
    def velocity(*args):
        return {'vp': vp, 'vs': vs}
    return velocity


LAYER_KIND = {
    'iso': _iso_model,
    'ani': None,
}


class Layer(object):
    def __init__(self, kind='iso', name='flat', *args, **kwargs):
        self.kind = kind
        self.top = Horizon(**kwargs)
        self.units = Units(**kwargs)
        self.name = name
        self.predict = None
        self.args = args
        self.kwargs = kwargs
        self.fit(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.predict = LAYER_KIND[self.kind](*args, **kwargs)

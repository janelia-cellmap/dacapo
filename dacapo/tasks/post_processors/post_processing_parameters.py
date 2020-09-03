class PostProcessingParameters:

    def __init__(self, id_=0, **kwargs):
        self.id = id_
        self.values = kwargs

    def __getitem__(self, key):
        return self.values[key]

    def __getattr__(self, attr):
        return self.values[attr]

    def items(self):
        return self.values.items()

    def to_dict(self):
        return dict(self.values)

    def __repr__(self):
        return f'({self.id}) ' + ':'.join(f'{k}={v}' for k, v in self.items())

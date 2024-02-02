class Immutable:
    def __init__(self):
        self._is_immutable = False
    def make_immutable(self):
        self._is_immutable = True
    def __setattr__(self, key, value):
        if hasattr(self, '_is_immutable') and self._is_immutable and key != '_is_immutable':
            raise AttributeError("Cannot modify immutable object")
        super().__setattr__(key, value)
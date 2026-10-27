class SafeAttr:
    def __init__(self, value):
        self._value = value

    def get_attr(self, name, default=None):
        if self._value is None:
            return SafeAttr(None)
        # Wrap the attribute in SafeAttr to support chained access.
        return SafeAttr(getattr(self._value, name, default))

    def get(self):
        """Return the final value."""
        return self._value

    def __repr__(self):
        return f"SafeAttr({self._value!r})"

class ExceptionVariable:
    """Raises the exception when an attribute is accessed on this variable."""
    def __init__(self, e):
        self.e = e

    def __getattr__(self, name):
        raise self.e

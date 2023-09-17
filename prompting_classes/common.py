converter_registry = {}

def register_class(cls):
    converter_registry[cls.__name__] = cls
    return cls

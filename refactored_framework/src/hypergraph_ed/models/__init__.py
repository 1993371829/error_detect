MODEL_REGISTRY = {}


def register_model(name):
    def decorate(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"duplicate model: {name}")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorate


def build_model(cfg):
    from . import network  # Explicit registration, not arbitrary dynamic imports.
    if cfg.name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model: {cfg.name}")
    return MODEL_REGISTRY[cfg.name](cfg)

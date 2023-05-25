from ...utils.builder import HEAD, build_from_cfg


def build_head(cfg, **kwargs):
    return build_from_cfg(cfg, HEAD, **kwargs)
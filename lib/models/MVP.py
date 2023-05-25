from ..utils.builder import MODEL
from .PETR import PETRMultiView


@MODEL.register_module()
class MVP(PETRMultiView):

    def __init__(self, cfg):
        super(MVP, self).__init__(cfg)

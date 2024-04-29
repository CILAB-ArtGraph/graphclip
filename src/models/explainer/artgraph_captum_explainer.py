from torch_geometric.explain import CaptumExplainer


class ArtGraphCaptumExplainer(CaptumExplainer):
    SUPPORTED_METHODS = CaptumExplainer.SUPPORTED_METHODS
    SUPPORTED_METHODS.append("GraphIntegratedGradients")

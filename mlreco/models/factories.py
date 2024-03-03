import torch

def model_dict():
    """
    Returns dictionary of model classes using name keys (strings).

    Returns
    -------
    dict
    """

    from . import full_chain
    from . import uresnet
    from . import uresnet_ppn_chain
    from . import singlep
    from . import multip
    from . import spice
    from . import graph_spice
    from . import grappa
    from . import bayes_uresnet

    from . import vertex
    from . import file_io

    # Make some models available (not all of them, e.g. PPN is not standalone)
    models = {
        # Full reconstruction chain, including an option for deghosting
        "full_chain": (full_chain.FullChain, full_chain.FullChainLoss),
        # UresNet
        "uresnet": (uresnet.UResNet_Chain, uresnet.SegmentationLoss),
        # UResNet + PPN
        'uresnet_ppn_chain': (uresnet_ppn_chain.UResNetPPN, uresnet_ppn_chain.UResNetPPNLoss),
        # Single Particle Classifier
        "singlep": (singlep.ParticleImageClassifier, singlep.ParticleTypeLoss),
        # Multi Particle Classifier
        "multip": (singlep.MultiParticleImageClassifier, singlep.MultiParticleTypeLoss),
        # ParticleNet
        "particlenet": (multip.ParticleNet, multip.ParticleNetLoss),
        # SPICE
        "spice": (spice.SPICE, spice.SPICELoss),
        # Graph SPICE
        "graph_spice": (graph_spice.GraphSPICE, graph_spice.GraphSPICELoss),
        # Graph neural network Particle Aggregation (GrapPA)
        "grappa": (grappa.GNN, grappa.GNNLoss),
        # Bayesian UResNet
        "bayesian_uresnet": (bayes_uresnet.BayesianUResNet, bayes_uresnet.SegmentationLoss),
        # DUQ UResNet
        "duq_uresnet": (bayes_uresnet.DUQUResNet, bayes_uresnet.DUQSegmentationLoss),
        # Vertex PPN
        'vertex_ppn': (vertex.VertexPPNChain, vertex.UResNetVertexLoss),
        # Vertex Pointnet
        'vertex_pointnet': (vertex.VertexPointNet, vertex.VertexPointNetLoss),
        # File I/O placeholder
        'file_io': (file_io.FileIOPlaceHolder, file_io.FileIOPlaceHolderLoss),
    }
    return models


def construct(name):
    """
    Returns an instance of a model class based on its name key (string).

    Parameters
    ----------
    name: str
        Key for the model. See source code for list of available models.

    Returns
    -------
    object
    """
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided: %s" % name)
    return models[name]

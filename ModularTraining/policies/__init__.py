from .mlp import MLPActorCritic
from .edge_transformer import EdgeTransformerPolicy
from .edge_transformer_deg import EdgeTransformerDegPolicy
from .edge_transformer_nocn import EdgeTransformerNoCNPolicy
from .mlp_featured import MLPFeaturedPolicy

POLICIES = {
    "mlp": MLPActorCritic,
    "edge_transformer": EdgeTransformerPolicy,
    "edge_transformer_deg": EdgeTransformerDegPolicy,
    "edge_transformer_nocn": EdgeTransformerNoCNPolicy,
    "mlp_featured": MLPFeaturedPolicy,
}


def make_policy(name, **kwargs):
    """Instantiate a policy by name. Extra kwargs forwarded to constructor."""
    if name not in POLICIES:
        raise ValueError(f"Unknown policy '{name}'. Choose from: {list(POLICIES.keys())}")
    return POLICIES[name](**kwargs)

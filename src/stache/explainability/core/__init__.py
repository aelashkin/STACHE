"""Domain-neutral robustness-region and counterfactual contracts."""

from .connector import (
    ConnectorIdentity,
    DiscreteActionSpec,
    ExactActionInvariance,
    FormalDistanceLayer,
    MetricCertificate,
    ObservationSpec,
    PolicyConnector,
    SearchConnector,
)
from .policy import (
    ActionCacheRecord,
    ActionOracle,
    ActionShapeError,
    ActionValidationError,
    CacheRestoreError,
    ModelActionOracle,
    ModelCompatibilityError,
    OracleStats,
    TableActionOracle,
    TableThenModelActionOracle,
    UnknownTableKeyError,
    normalize_discrete_action,
)

__all__ = [
    "ActionCacheRecord",
    "ActionOracle",
    "ActionShapeError",
    "ActionValidationError",
    "CacheRestoreError",
    "ConnectorIdentity",
    "DiscreteActionSpec",
    "ExactActionInvariance",
    "FormalDistanceLayer",
    "MetricCertificate",
    "ModelActionOracle",
    "ModelCompatibilityError",
    "ObservationSpec",
    "OracleStats",
    "PolicyConnector",
    "SearchConnector",
    "TableActionOracle",
    "TableThenModelActionOracle",
    "UnknownTableKeyError",
    "normalize_discrete_action",
]

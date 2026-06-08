# Lazy imports: trainer.py and adkfift_trainer.py require torchmetrics which may
# not be installed in all environments. GNN_Encoder (used by AttFPGNN) does not
# need them, so we defer import to avoid hard crash at module load time.
# adkf_model.py also requires gpytorch — same lazy treatment.
try:
    from .trainer import Meta_Trainer
    from .adkfift_trainer import ADKF_Meta_Trainer
    from .adkf_model import ADKFModel
except ImportError:
    pass  # torchmetrics/gpytorch not installed; these classes are unavailable

from .mol_model import ContextAwareRelationNet
"""purediscover — pure Python symbolic discovery."""
from purediscover.symbolic import discover_law, Grammar
from purediscover.groups import explore_groups
from purediscover.mdl import mdl_score
from purediscover.audit import replay_audit
__all__ = ["discover_law", "Grammar", "explore_groups", "mdl_score", "replay_audit"]
__version__ = "0.1.0"

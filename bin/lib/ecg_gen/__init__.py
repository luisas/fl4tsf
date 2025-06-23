"""
ECG generation package for federated learning research.
Generates clinically realistic, non-iid ECG data across multiple client archetypes.
"""

from .generator import generate_clients

__version__ = "0.1.0"
__all__ = ["generate_clients"]
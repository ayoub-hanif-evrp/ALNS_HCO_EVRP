"""Shared instance types. Both benchmarks use the same converted-folder format."""
from src.schneider.data.instance_loader import EVRPTWInstance, Node, VehicleParams, load_instance

__all__ = ["EVRPTWInstance", "Node", "VehicleParams", "load_instance"]

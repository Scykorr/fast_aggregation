"""Определение CPU, RAM, GPU и доступных бэкендов llama.cpp."""

from fast_aggregation.resources.probe import apply_vram_override, probe_machine

__all__ = ["apply_vram_override", "probe_machine"]

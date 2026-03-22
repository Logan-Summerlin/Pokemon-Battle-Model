"""Data pipeline for Pokemon battle model training.

Phase 2 modules:
- replay_parser: Parse Metamon-format replay files
- observation: Build first-person observations from parsed replays
- tensorizer: Convert observations to fixed-size tensors
- dataset: PyTorch datasets and data loading utilities
- priors: Metagame usage statistics for soft opponent prediction
"""

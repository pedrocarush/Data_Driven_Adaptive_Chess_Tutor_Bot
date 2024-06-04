#!/bin/bash
set -e
python3 1_filter_pgn_optimized.py
python3 2_pgn_to_csv_optimized.py
python3 3_csv_parser.py
python3 4_process_moves.py
python3 5_extract_features.py
python3 6_model_generation.py
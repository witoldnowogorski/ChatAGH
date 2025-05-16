import os
import sys

tests_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(tests_dir)
sys.path.insert(0, project_root)
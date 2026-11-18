#!/usr/bin/env python3
import os, runpy, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(ROOT, 'harness')
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)
runpy.run_path(os.path.join(HARNESS, 'serve.py'), run_name='__main__')

import sys
import os

# add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.object_boundary.boundary import main

if __name__ == "__main__":
    main()

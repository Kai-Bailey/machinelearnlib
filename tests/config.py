import os
import sys

# Insert the path to the machinelearnlib folder so python will search those modules on import statements
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import machinelearnlib

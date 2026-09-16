"""
Make the modules under scripts/ importable in tests. The editable install only exposes the
src package, so the project root has to be on the path for tests of the analysis scripts.
"""

import sys

from pyprojroot import here

sys.path.insert(0, str(here()))

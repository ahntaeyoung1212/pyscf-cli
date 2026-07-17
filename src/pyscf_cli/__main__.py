"""Support `python -m pyscf_cli` as an alias for the pyscf-cli command."""

import sys

from .main import main

sys.exit(main())

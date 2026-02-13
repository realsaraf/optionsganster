"""
Run the OptionsGanster application
"""
import os, sys

# Ensure cwd is the directory containing this script (src/)
# so that `app` package is importable regardless of where we're invoked from.
_here = os.path.dirname(os.path.abspath(__file__))
os.chdir(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)

import uvicorn
from app.config import settings

if __name__ == "__main__":
    is_dev = settings.ENVIRONMENT == "development"

    print(f"""
    ╔═══════════════════════════════════════════╗
    ║        OptionsGanster v2.0                ║
    ║        VPA Options Analysis Tool          ║
    ╠═══════════════════════════════════════════╣
    ║  Server starting at:                      ║
    ║  http://{settings.HOST}:{settings.PORT}                   ║
    ╚═══════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=is_dev
    )

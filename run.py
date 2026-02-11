"""
Run the OptionsGanster application
"""
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

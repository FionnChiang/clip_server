import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import server_config
from .database import init_db, close_db
from .routers import projects, datasets, training, models, inference, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Layout Classifier Platform",
        description="Document Layout Classification Training & Inference Platform",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def on_startup():
        try:
            await init_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}. Some features may not be available.")

    @app.on_event("shutdown")
    async def on_shutdown():
        await close_db()

    app.include_router(settings.router)
    app.include_router(projects.router)
    app.include_router(datasets.router)
    app.include_router(training.router)
    app.include_router(models.router)
    app.include_router(inference.router)

    frontend_dist = server_config.frontend_dist
    if not os.path.isabs(frontend_dist):
        frontend_dist = str((Path(__file__).resolve().parent.parent / frontend_dist).resolve())
    if os.path.isdir(frontend_dist):
        app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str = ""):
            file_path = os.path.join(frontend_dist, "index.html")
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            return {"message": "Frontend not built. Run 'npm run build' in frontend directory."}

    return app


app = create_app()

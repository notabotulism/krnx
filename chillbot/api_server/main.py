"""
KRNX API Server

FastAPI application exposing the KRNX kernel and fabric.

Usage:
    uvicorn api_server.main:app --host 0.0.0.0 --port 6380 --reload

Or programmatically:
    from api_server.main import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6380)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from .config import get_settings
from .deps import init_dependencies, shutdown_dependencies
from .routes import basic_router, advanced_router
from .schemas import ErrorResponse, ErrorDetail, ResponseMeta


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown.
    
    Initializes the controller and fabric on startup,
    and gracefully shuts them down on exit.
    """
    settings = get_settings()
    
    # Startup
    print("=" * 60)
    print("KRNX API Server Starting...")
    print("=" * 60)
    
    try:
        init_dependencies(settings)
        print(f"[OK] API server ready at http://{settings.host}:{settings.port}")
        print("=" * 60)
        yield
    finally:
        # Shutdown
        print("=" * 60)
        print("KRNX API Server Shutting Down...")
        print("=" * 60)
        shutdown_dependencies()
        print("[OK] Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="KRNX API",
    description="""
    Temporal Memory Infrastructure for AI Agents.
    
    KRNX provides:
    - **Event Storage**: Append-only event log with temporal indexing
    - **Temporal Replay**: Reconstruct state at any point in time (THE DIFFERENTIATOR)
    - **Provenance Tracking**: Cryptographic hash chains for auditability
    - **Supersession Detection**: Track fact versions and contradictions
    - **Multi-Agent Coordination**: Consumer groups and event streams
    - **Workflow Branches**: Explore alternative memory states
    
    ## API Organization
    
    **Basic API** (`/api/v1/`): Core CRUD and temporal operations
    - Events: Create, read, list, delete
    - Temporal: State reconstruction, replay, timeline
    - Health: System status and metrics
    
    **Advanced API** (`/api/v1/`): Full playbook features
    - Provenance: Hash chain verification
    - Supersession: Fact versioning
    - Context: LLM-ready memory retrieval
    - Agents: Multi-agent coordination
    - Branches: Workflow branching/merging
    """,
    version="0.3.10",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    """
    error_type = type(exc).__name__
    
    # Map known exceptions to appropriate responses
    if "BackpressureError" in error_type:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error=ErrorDetail(
                    type="https://krnx.dev/errors/backpressure",
                    code="BACKPRESSURE",
                    title="System Under Load",
                    detail="The system is under heavy load. Please retry.",
                    status=503
                ),
                meta=ResponseMeta()
            ).model_dump()
        )
    
    if "ValidationError" in error_type:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    type="https://krnx.dev/errors/validation",
                    code="VALIDATION_ERROR",
                    title="Validation Error",
                    detail=str(exc),
                    status=400
                ),
                meta=ResponseMeta()
            ).model_dump()
        )
    
    # Default internal error
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                type="https://krnx.dev/errors/internal",
                code="INTERNAL_ERROR",
                title="Internal Server Error",
                detail=str(exc),
                status=500
            ),
            meta=ResponseMeta()
        ).model_dump()
    )


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Response-Time header to all responses."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    return response


# Include routers
api_prefix = settings.api_prefix

app.include_router(
    basic_router,
    prefix=api_prefix,
    tags=["Basic"]
)

app.include_router(
    advanced_router,
    prefix=api_prefix,
    tags=["Advanced"]
)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "KRNX API",
        "version": "0.3.10",
        "description": "Temporal Memory Infrastructure for AI Agents",
        "docs": "/docs",
        "health": f"{api_prefix}/health",
    }


# CLI runner
def main():
    """Run the server from command line."""
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "api_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()

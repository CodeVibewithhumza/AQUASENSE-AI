"""Custom FastAPI middleware and exception handlers for standardized error responses."""
import time
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import logger


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Logs incoming request method/path and calculates processing duration."""
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000  # ms
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

        # Skip health check logging to prevent log pollution
        if request.url.path != "/health":
            logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} ({process_time:.2f}ms)")

        return response


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Formats 422 Request Validation errors into clean, structured JSON."""
    errors = []
    for err in exc.errors():
        field = " -> ".join([str(loc) for loc in err.get("loc", [])])
        msg = err.get("msg", "Invalid value")
        errors.append({"field": field, "message": msg, "type": err.get("type")})

    logger.warning(f"Validation error on {request.url.path}: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Input validation failed. Please check your parameter values.",
            "details": errors
        }
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches unhandled server exceptions and returns standard 500 JSON response."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred during processing. Please verify server logs.",
            "details": str(exc)
        }
    )

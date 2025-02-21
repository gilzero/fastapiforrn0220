# filepath: main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import time
from dotenv import load_dotenv
import os
import uvicorn
from models import ChatRequest, HealthResponse
from aiproviders import (
    stream_response, PROVIDER_MODELS, SUPPORTED_PROVIDERS,
    client, anthropic_client, genai_client, check_provider_health
)
from logging_config import logger, debug_with_context
import traceback

# load env variables
load_dotenv()

PORT = os.getenv("PORT")

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request timing and details"""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    debug_with_context(logger,
        f"Request completed: {request.method} {request.url.path}",
        duration=f"{duration:.3f}s",
        status_code=response.status_code,
        client_host=request.client.host
    )
    return response

# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Overall system health check"""
    logger.info("Health check endpoint called")
    try:
        return {"status": "OK", "message": "System operational"}
    except Exception as e:
        logger.error("Health check failed", exc_info=True)
        return {
            "status": "ERROR",
            "error": {"message": str(e)}
        }

@app.get("/health/{provider}")
async def provider_health_check(provider: str):
    """Check health of a specific provider."""
    logger.info(f"Provider health check called for: {provider}")
    
    # First validate the provider
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Invalid provider. Supported providers are: {', '.join(SUPPORTED_PROVIDERS)}")
    
    try:
        success, message, duration = await check_provider_health(provider)
        
        # Fix the debug_with_context call
        debug_with_context(
            logger,
            f"Health check completed for {provider}",  # message as first arg
            success=success,
            duration=f"{duration:.3f}s"
        )
        
        return {
            "provider": provider,
            "status": "OK" if success else "ERROR",  # Changed from "healthy"/"unhealthy" to "OK"/"ERROR"
            "message": message,
            "metrics": {
                "responseTime": f"{duration:.3f}s"
            },
            "error": {"message": message} if not success else None
        }
    except Exception as e:
        logger.error(f"Health check failed for provider {provider}")
        logger.error(traceback.format_exc())
        return {
            "provider": provider,
            "status": "ERROR",  # Changed from "error" to "ERROR"
            "error": {"message": str(e)},
            "metrics": {
                "responseTime": "N/A"
            }
        }

# Chat endpoint
@app.post("/chat/{provider}")
async def chat(provider: str, request: ChatRequest):
    """Stream chat responses from an AI provider"""
    logger.debug("Chat endpoint called",
        extra={
            "context": {
                "provider": provider,
                "message_count": len(request.messages),
                "validation_state": "post-fastapi-validation"
            }
        }
    )
    start_time = time.time()

    try:
        if provider not in SUPPORTED_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider. Supported providers are: {', '.join(SUPPORTED_PROVIDERS)}"
            )

        debug_with_context(logger,
            f"Chat request received for provider: {provider}",
            message_count=len(request.messages),
            first_message_role=request.messages[0].role if request.messages else None
        )

        response = StreamingResponse(
            stream_response(request, provider),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

        init_duration = time.time() - start_time
        debug_with_context(logger,
            "Chat stream response initialized",
            init_duration=f"{init_duration:.3f}s",
            provider=provider
        )
        return response

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error with {provider}: {str(e)}")

if __name__ == "__main__":
    port = int(PORT) if PORT else 3050
    uvicorn.run(app, host="0.0.0.0", port=port)
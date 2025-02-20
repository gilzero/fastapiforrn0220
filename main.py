# filepath: pyserver/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import time
import asyncio
from datetime import datetime, UTC
from dotenv import load_dotenv
import os
import uvicorn
from models import ChatRequest, HealthResponse
from aiproviders import stream_response, MODELS
from logging_config import logger

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


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "OK", "message": "System operational"}


@app.get("/health/{provider}", response_model=HealthResponse)
async def provider_health_check(provider: str):
    logger.info(f"Provider health check called for: {provider}")
    if provider not in MODELS:
        logger.error(f"Invalid provider requested: {provider}")
        raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")

    start_time = time.time()
    await asyncio.sleep(0.1)
    duration = time.time() - start_time

    logger.info(f"Provider {provider} health check completed in {duration:.2f}s")
    return {
        "status": "OK",
        "provider": provider,
        "metrics": {"responseTime": duration}
    }


# Chat endpoint
@app.post("/chat/{provider}")
async def chat(provider: str, request: ChatRequest):
    start_time = time.time()

    try:
        if provider not in MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")

        valid_models = [MODELS[provider]["default"], MODELS[provider]["fallback"]]
        if request.model not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model for {provider}. Valid models are: {', '.join(valid_models)}"
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

        duration = time.time() - start_time
        logger.info(f"Chat request initialized in {duration:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error with {provider}: {str(e)}")


if __name__ == "__main__":
    port = int(PORT) if PORT else 3050
    uvicorn.run(app, host="0.0.0.0", port=port)
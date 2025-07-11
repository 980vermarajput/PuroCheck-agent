"""
PuroCheck AI API - FastAPI Application

This module provides REST API endpoints for the PuroCheck AI eligibility agent.
Supports document upload and streaming evaluation results.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import json
import os
import asyncio
import tempfile
import shutil
import time
from pathlib import Path
import logging

# Import your existing modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import PuroCheckAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PuroCheck AI API",
    description="AI-powered biochar project eligibility evaluation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
agent_instance = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup"""
    global agent_instance
    logger.info("Starting PuroCheck AI API...")
    # We'll initialize the agent when needed to avoid startup delays

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down PuroCheck AI API...")

@app.get("/")
async def root():
    """Root endpoint redirects to the stream tester UI"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/stream-tester")

@app.get("/api-info")
async def api_info():
    """API information endpoint"""
    return {
        "message": "PuroCheck AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker healthcheck"""
    return {"status": "healthy", "service": "purocheck-ai"}

@app.post("/evaluate")
async def evaluate_documents(
    files: List[UploadFile] = File(...),
    registry: str = Query("puro"),
    api_provider: str = Query("auto")
):
    """
    Evaluate uploaded documents against registry checklist with streaming results
    
    Args:
        files: List of uploaded PDF/document files
        registry: Registry type (puro, verra, etc.)
        api_provider: AI provider (auto, openai, groq)
    
    Returns:
        StreamingResponse with evaluation progress and results
    """
    
    # Debug logging for parameters
    logger.info("====================== REQUEST PARAMETERS ======================")
    logger.info(f"RAW PARAMETERS RECEIVED - Registry: '{registry}', API Provider: '{api_provider}'")
    logger.info(f"Parameter types - Registry: {type(registry)}, API Provider: {type(api_provider)}")
    logger.info(f"Files received: {[f.filename for f in files]}")
    logger.info("===============================================================")
    
    # Ensure parameters are properly processed
    registry = registry.strip().lower() if registry else "puro"
    api_provider = api_provider.strip().lower() if api_provider else "auto"
    
    logger.info(f"PROCESSED PARAMETERS - Registry: '{registry}', API Provider: '{api_provider}'")
    # Validate inputs
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    if registry.lower() not in ["puro", "verra"]:
        raise HTTPException(status_code=400, detail="Unsupported registry. Use 'puro' or 'verra'")
    
    # Validate file types
    allowed_extensions = {'.pdf', '.txt', '.docx', '.xlsx', '.xls','.csv'}
    for file in files:
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.filename}. Allowed: {', '.join(allowed_extensions)}"
            )
    
    # Read all files immediately to avoid I/O issues in streaming
    file_data = []
    for file in files:
        content = await file.read()
        file_data.append({
            'filename': file.filename,
            'content': content
        })

    async def stream_evaluation():
        temp_dir = None
        try:
            # Create temporary directory for this evaluation
            temp_dir = tempfile.mkdtemp(prefix="purocheck_")
            logger.info(f"Created temp directory: {temp_dir}")
            
            # 1. Initial status
            yield f"data: {json.dumps({'status': 'starting', 'message': 'Starting evaluation process...'})}\n\n"
            
            # 2. Save uploaded files
            yield f"data: {json.dumps({'status': 'uploading', 'message': f'Saving {len(file_data)} uploaded files...'})}\n\n"
            
            saved_files = []
            for i, file_info in enumerate(file_data):
                try:
                    file_path = os.path.join(temp_dir, file_info['filename'])
                    
                    with open(file_path, "wb") as buffer:
                        buffer.write(file_info['content'])
                    saved_files.append(file_path)
                    
                    filename = file_info['filename']
                    message = f'Saved {filename} ({i+1}/{len(file_data)})'
                    yield f"data: {json.dumps({'status': 'uploading', 'message': message})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Error saving file {file_info['filename']}: {str(e)}")
                    raise ValueError(f"Failed to save file {file_info['filename']}: {str(e)}")
            
            # 3. Initialize agent
            yield f"data: {json.dumps({'status': 'initializing', 'message': 'Initializing PuroCheck agent...'})}\n\n"
            # Create a unique vector store directory for this evaluation
            eval_vector_store_dir = os.path.join(temp_dir, "vector_store")
            os.makedirs(eval_vector_store_dir, exist_ok=True)
              # Debug log to check parameter values
            logger.info("====================== AGENT INITIALIZATION ======================")
            logger.info(f"USING PARAMETERS - Registry: '{registry}', API Provider: '{api_provider}'")
            logger.info(f"Vector Store Dir: '{eval_vector_store_dir}'")
            logger.info("=================================================================")
            
            # Determine the correct checklist path based on registry
            if registry.lower() == "verra":
                checklist_path = "checklist/verra_vm0047_checklist.json"
            else:
                checklist_path = f"checklist/{registry.lower()}_biochar_checklist.json"
            
            agent = PuroCheckAgent(
                data_dir=temp_dir,
                checklist_path=checklist_path,
                api_provider=api_provider,
                registry=registry.lower(),
                vector_store_dir=eval_vector_store_dir,
                force_rebuild_vectorstore=True, 
                model_name="gpt-4.1-nano" if api_provider=='openai' else None  # Use appropriate model,
            )
            
            # 4. Process documents
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Processing documents and creating vector store...'})}\n\n"
            
            # Initialize the agent (this processes documents)
            try:
                logger.info(f"Initializing agent with {len(saved_files)} documents from temp directory: {temp_dir}")
                await asyncio.to_thread(agent.initialize)
                logger.info("Agent initialization successful")
            except Exception as e:
                logger.error(f"Agent initialization failed: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'status': 'error', 'message': f'Document processing failed: {str(e)}'})}\n\n"
                raise
            
            # 5. Run the full evaluation using the agent's built-in method
            yield f"data: {json.dumps({'status': 'evaluating', 'message': 'Running project evaluation...'})}\n\n"
            
            # Define a progress callback to stream updates
            async def progress_callback(update):
                event_type = update.get('type', 'progress')
                
                if event_type == 'section_start':
                    message = f"Evaluating section: {update['section_title']}"
                    yield f"data: {json.dumps({'status': 'section_start', 'message': message, 'section': update['section_title']})}\n\n"
                
                elif event_type == 'item_start':
                    message = f"Evaluating: {update['requirement']}"
                    yield f"data: {json.dumps({'status': 'item_start', 'message': message, 'requirement': update['requirement']})}\n\n"
                
                elif event_type == 'item_complete':
                    yield f"data: {json.dumps({'status': 'item_complete', 'result': {
                        'requirement': update['requirement'],
                        'status': update['status'],
                        'reason': update['reason'],
                        'evidence_found': update['evidence_found'],
                        'missing_evidence': update.get('missing_evidence', []),
                        'confidence_score': update.get('confidence_score', 0)
                    }, 'summary': update['summary']})}\n\n"
            
            # Create a queue for progress updates and results
            progress_queue = asyncio.Queue()
            
            # Get the current event loop
            loop = asyncio.get_running_loop()
            
            # Callback that directly formats and returns the SSE event data
            def format_update_as_sse(update):
                event_type = update.get('type', 'progress')
                
                if event_type == 'section_start':
                    message = f"Evaluating section: {update['section_title']}"
                    return f"data: {json.dumps({'status': 'section_start', 'message': message, 'section': update['section_title']})}\n\n"
                
                elif event_type == 'item_start':
                    message = f"Evaluating: {update['requirement']}"
                    return f"data: {json.dumps({'status': 'item_start', 'message': message, 'requirement': update['requirement']})}\n\n"
                
                elif event_type == 'item_complete':
                    return f"data: {json.dumps({'status': 'item_complete', 'result': {
                        'requirement': update['requirement'],
                        'status': update['status'],
                        'reason': update['reason'],
                        'evidence_found': update['evidence_found'],
                        'missing_evidence': update.get('missing_evidence', []),
                        'confidence_score': update.get('confidence_score', 0)
                    }, 'summary': update['summary']})}\n\n"
                
                return None
            
            # Callback that immediately queues formatted SSE messages
            def queue_callback(update):
                sse_data = format_update_as_sse(update)
                if sse_data:
                    loop.call_soon_threadsafe(lambda: asyncio.create_task(progress_queue.put(sse_data)))
            
            # Start evaluation in a separate task
            evaluation_task = asyncio.create_task(
                asyncio.to_thread(agent.evaluate_project, queue_callback)
            )
            
            # Stream progress updates while evaluation is running
            last_heartbeat = time.time()
            heartbeat_interval = 2.0  # Send heartbeat every 2 seconds
            
            while True:
                if evaluation_task.done():
                    # Get the results
                    try:
                        results = evaluation_task.result()
                        # Send final results
                        yield f"data: {json.dumps({'status': 'complete', 'message': 'Evaluation finished successfully!', 'results': results})}\n\n"
                    except Exception as e:
                        logger.error(f"Error getting evaluation results: {str(e)}")
                        yield f"data: {json.dumps({'status': 'error', 'message': f'Evaluation error: {str(e)}'})}\n\n"
                    break
                
                # Check if there are any progress updates in the queue
                try:
                    # Get direct SSE-formatted message from queue with a short timeout
                    sse_message = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    # Directly yield it (already formatted as SSE)
                    yield sse_message
                    # Reset heartbeat timer
                    last_heartbeat = time.time()
                    
                except asyncio.TimeoutError:
                    # No updates in the queue, check if we need to send a heartbeat
                    current_time = time.time()
                    if current_time - last_heartbeat > heartbeat_interval:
                        # Send heartbeat to keep connection alive
                        yield f"data: {json.dumps({'status': 'heartbeat', 'message': 'Still processing...', 'timestamp': current_time})}\n\n"
                        last_heartbeat = current_time
                    
                    # Small sleep to avoid CPU spinning
                    await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'message': f'Evaluation failed: {str(e)}'})}\n\n"
        
        finally:
            # Cleanup temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
    
    return StreamingResponse(
        stream_evaluation(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/registries")
async def get_supported_registries():
    """Get list of supported registries"""
    return {
        "registries": [
            {
                "name": "puro",
                "display_name": "Puro.earth",
                "description": "Puro.earth Biochar Methodology",
                "checklist_file": "puro_biochar_checklist.json"
            },
            {
                "name": "verra",
                "display_name": "Verra VCS",
                "description": "Verra VM0047 Biochar Methodology",
                "checklist_file": "verra_vm0047_checklist.json"
            }
        ]
    }

@app.get("/checklist/{registry}")
async def get_checklist(registry: str):
    """Get the checklist for a specific registry"""
    if registry.lower() not in ["puro", "verra"]:
        raise HTTPException(status_code=404, detail="Registry not found")
    
    try:
        # Determine the correct checklist path based on registry
        if registry.lower() == "verra":
            checklist_path = "checklist/verra_vm0047_checklist.json"
        else:
            checklist_path = f"checklist/{registry.lower()}_biochar_checklist.json"
            
        with open(checklist_path, 'r') as f:
            checklist = json.load(f)
        return {
            "registry": registry.lower(),
            "items": len(checklist),
            "checklist": checklist
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Checklist for {registry} not found")

@app.get("/stream-tester", response_class=HTMLResponse)
async def get_stream_tester():
    """Get the HTML stream tester page"""
    try:
        with open("stream_tester.html", "r") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Stream tester page not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# 🐳 PuroCheck AI - Docker Setup

This guide explains how to run PuroCheck AI using Docker for easy deployment and API access.

## 📋 Prerequisites

- Docker and Docker Compose installed
- OpenAI API key or Groq API key

## 🚀 Quick Start

### 1. Setup Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# Required: Either OPENAI_API_KEY or GROQ_API_KEY
```

### 2. Start the Application

```bash
# Option 1: Use the convenience script
./start.sh

# Option 2: Manual Docker Compose
docker-compose up -d
```

### 3. Verify Installation

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Status**: http://localhost:8000/

## 📡 API Usage

### Streaming Evaluation Endpoint

**POST** `/evaluate`

Upload documents and get real-time evaluation results:

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -F "files=@your_document.pdf" \
  -F "registry=puro" \
  --no-buffer
```

### Get Supported Registries

**GET** `/registries`

```bash
curl http://localhost:8000/registries
```

### Get Checklist for Registry

**GET** `/checklist/{registry}`

```bash
curl http://localhost:8000/checklist/puro
```

## 🔧 Development

### Run in Development Mode

```bash
# Start with auto-reload
docker-compose up --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using the Python Client

```bash
# Install requests if not already installed
pip install requests

# Use the example client
python client_example.py data/your_document.pdf
```

## 📁 File Structure

```
.
├── api/                 # FastAPI application
│   ├── main.py         # Main API endpoints
│   └── __init__.py
├── data/               # Document upload directory
├── chroma_db/          # Vector database storage
├── checklist/          # Registry checklists
├── docker-compose.yml  # Docker services
├── Dockerfile          # Container definition
├── .env.example        # Environment template
├── start.sh           # Convenience startup script
└── client_example.py  # Example API client
```

## 🌐 API Features

### ✅ Implemented

- **Document Upload**: Support for PDF, TXT, DOCX files
- **Streaming Evaluation**: Real-time progress updates
- **Multi-Registry**: Puro.earth and Verra VCS support
- **Health Checks**: Container health monitoring
- **CORS Support**: Cross-origin requests enabled

### 🔄 Stream Response Format

The `/evaluate` endpoint returns Server-Sent Events (SSE) with JSON data:

```javascript
// Example stream events
data: {"status": "starting", "message": "Starting evaluation process..."}
data: {"status": "uploading", "message": "Saving uploaded files..."}
data: {"status": "processing", "message": "Processing documents..."}
data: {"status": "evaluating", "current_item": 1, "total_items": 25, "item_name": "H/C ratio requirement"}
data: {"status": "item_complete", "item_index": 0, "result": {...}, "summary": {...}}
data: {"status": "complete", "message": "Evaluation finished!", "final_summary": {...}}
```

## 🛠️ Troubleshooting

### Common Issues

1. **API Keys Not Set**

   ```bash
   # Check your .env file
   cat .env
   ```

2. **Port Already in Use**

   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8001:8000"  # Use different external port
   ```

3. **Memory Issues**

   ```bash
   # Increase Docker memory limit or use Groq instead of OpenAI
   # In .env file, ensure GROQ_API_KEY is set
   ```

4. **Permission Issues**
   ```bash
   # Fix directory permissions
   sudo chown -R $USER:$USER data/ chroma_db/
   ```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f purocheck-api
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Rebuild and restart
docker-compose up --build -d
```

## 📊 Monitoring

### Health Checks

The API includes built-in health checks:

```bash
# Check if API is healthy
curl http://localhost:8000/health

# Expected response
{"status": "healthy", "service": "purocheck-ai"}
```

### Performance

- **Memory Usage**: ~1-2GB depending on document size
- **Evaluation Time**: ~30 seconds to 5 minutes per document set
- **Concurrency**: Single evaluation at a time (can be enhanced)

## 🔒 Security Notes

For production deployment:

1. **Environment Variables**: Use secure secret management
2. **CORS**: Configure specific allowed origins
3. **File Upload**: Add file size and type validation
4. **Rate Limiting**: Implement request rate limiting
5. **Authentication**: Add API authentication if needed

## 📞 Support

- Check logs: `docker-compose logs -f`
- Restart services: `docker-compose restart`
- Clean restart: `docker-compose down && docker-compose up -d`

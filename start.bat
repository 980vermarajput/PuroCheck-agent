@echo off
REM PuroCheck AI Docker Setup and Run Script for Windows

echo 🚀 Setting up PuroCheck AI with Docker...

REM Check if Docker is installed
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Docker is not installed. Please install Docker first.
    exit /b 1
)

REM Check if Docker Compose is installed
where docker-compose >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ⚠️  Please edit .env file and add your API keys before running the application.
    echo    Required: OPENAI_API_KEY or GROQ_API_KEY
    echo Press Enter after you've updated the .env file...
    pause >nul
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist data mkdir data
if not exist chroma_db mkdir chroma_db

REM Build and start the application
echo 🔨 Building Docker image...
docker-compose build

echo 🚀 Starting PuroCheck AI API...
docker-compose up -d

REM Wait for the service to be healthy
echo ⏳ Waiting for service to be ready...
timeout /t 10 /nobreak >nul

REM Check if the service is running
curl -f http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✅ PuroCheck AI API is running successfully!
    echo.
    echo 🌐 API Documentation: http://localhost:8000/docs
    echo 🔍 API Health Check: http://localhost:8000/health
    echo 📊 API Status: http://localhost:8000/
    echo.
    echo 📝 To view logs: docker-compose logs -f
    echo 🛑 To stop: docker-compose down
) else (
    echo ❌ Service failed to start. Check logs with: docker-compose logs
)

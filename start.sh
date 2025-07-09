#!/bin/bash

# PuroCheck AI Docker Setup and Run Script

echo "🚀 Setting up PuroCheck AI with Docker..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your API keys before running the application."
    echo "   Required: OPENAI_API_KEY or GROQ_API_KEY"
    read -p "Press Enter after you've updated the .env file..."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data chroma_db

# Build and start the application
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting PuroCheck AI API..."
docker-compose up -d

# Wait for the service to be healthy
echo "⏳ Waiting for service to be ready..."
sleep 10

# Check if the service is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ PuroCheck AI API is running successfully!"
    echo ""
    echo "🌐 API Documentation: http://localhost:8000/docs"
    echo "🔍 API Health Check: http://localhost:8000/health"
    echo "📊 API Status: http://localhost:8000/"
    echo ""
    echo "📝 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Service failed to start. Check logs with: docker-compose logs"
fi

#!/bin/bash
# Super simple build script - just build the image

echo "🏗️  Building pandas-mcp-server image..."
docker build -t pandas-mcp-server:latest .

if [ $? -eq 0 ]; then
    echo "✅ Build completed successfully!"
    echo ""
    echo "🚀 To run the server:"
    echo "   docker run -p 8000:8000 pandas-mcp-server:latest"
    echo ""
    echo "🔗 Connect to: http://localhost:8000/sse"
    echo ""
    echo "📦 Image ready: pandas-mcp-server:latest"
else
    echo "❌ Build failed!"
    exit 1
fi
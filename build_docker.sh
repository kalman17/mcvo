#!/bin/bash
MAX_ATTEMPTS=5
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "=========================================="
    echo "Docker build attempt $ATTEMPT of $MAX_ATTEMPTS"
    echo "=========================================="
    
    if docker build -t anycam-env .; then
        echo "✅ Build succeeded!"
        exit 0
    fi
    
    EXIT_CODE=$?
    echo "❌ Build failed with exit code $EXIT_CODE"
    
    # Check if it's a connection-related error
    if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then
        if docker build -t anycam-env . 2>&1 | grep -q -i "connection\|incomplete\|timeout\|network"; then
            echo "⚠️  Connection error detected. Retrying in 5 seconds..."
            sleep 5
            ATTEMPT=$((ATTEMPT + 1))
        else
            echo "❌ Non-connection error. Stopping retries."
            exit $EXIT_CODE
        fi
    else
        echo "❌ Max attempts reached. Build failed."
        exit $EXIT_CODE
    fi
done
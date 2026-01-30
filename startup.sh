#!/bin/bash

# Start the app using Gunicorn with Uvicorn worker
gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 src.api:app
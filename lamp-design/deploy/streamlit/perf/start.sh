#!/bin/bash
# Streamlit on a private port, nginx on $PORT, container dies if either does.
#
# bash, not sh: `wait -n` is a bash builtin and the base image's /bin/sh is
# dash, which fails with "Illegal option -n" and exits 2 before serving
# anything. Found by running the image.
set -e

PORT="${PORT:-8080}"
# nginx.conf ships with `listen 8080`; Cloud Run may hand us another port.
sed -i "s/listen 8080;/listen ${PORT};/" /etc/nginx/nginx.conf

streamlit run --server.port=8501 --server.address=127.0.0.1 \
    /app/deploy/streamlit/app.py &
STREAMLIT_PID=$!

nginx -g 'daemon off;' &
NGINX_PID=$!

# Whichever exits first takes the container with it, so an orchestrator
# restarts a half-dead pod instead of serving 502s behind a passing health
# check.
wait -n "$STREAMLIT_PID" "$NGINX_PID"
EXIT=$?
kill "$STREAMLIT_PID" "$NGINX_PID" 2>/dev/null || true
exit "$EXIT"

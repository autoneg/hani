#!/usr/bin/env bash
# Serve the public, no-auth HANI guest interface on port 5008 (the
# HANI_GUEST_PORT default), mounted at /hanguest behind nginx on
# study-host.example.org.
#
# `hani-guest` is a thin wrapper that sets HANI_GUEST_MODE=true and then
# execs `panel serve src/hani/app.py --port 5008 <extra args>`, so the
# extra Bokeh/panel flags below behave exactly the same way they do in
# run.sh for the authenticated app.

# Pick up --port=NNNN or --port NNNN from "$@" so we can whitelist the
# matching WebSocket origin. Without this, Bokeh rejects the browser's
# WS handshake on any non-default port and the page renders an empty
# Panel grid (HTML serves, content never streams in).
port=""
prev=""
for arg in "$@"; do
    case "$arg" in
        --port=*) port="${arg#--port=}" ;;
    esac
    if [ "$prev" = "--port" ]; then
        port="$arg"
    fi
    prev="$arg"
done

extra_origins=()
if [ -n "$port" ]; then
    extra_origins+=(--allow-websocket-origin "127.0.0.1:$port")
    extra_origins+=(--allow-websocket-origin "localhost:$port")
fi

hani-guest --prefix hanplay --index app --root-path=/hanplay \
    --allow-websocket-origin study-host.example.org \
    --allow-websocket-origin 127.0.0.1:5008 \
    --allow-websocket-origin localhost:5008 \
    --allow-websocket-origin 127.0.0.1:8000 \
    --allow-websocket-origin localhost:8000 \
    --session-token-expiration 900000 \
    "${extra_origins[@]}" \
    "$@"

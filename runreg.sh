#!/usr/bin/env bash

panel serve src/hani/register.py --port 5007 --prefix hanreg --allow-websocket-origin study-host.example.org --root-path=/hanreg --session-token-expiration 900000 "$@"

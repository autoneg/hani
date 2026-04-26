#!/bin/env bash

hani --cookie-secret my_super_safe_cookie_secret --dev --port 5006 --prefix hanapp --allow-websocket-origin anac.cs.brown.edu --root-path=/hanapp "$@"

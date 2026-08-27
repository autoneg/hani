#!/usr/bin/env bash
# Serve the authenticated HANI app on 127.0.0.1:5006, mounted at
# /hanapp behind nginx on study-host.example.org.
#
# The current CLI dropped the top-level `--cookie-secret`; the auth
# subcommand `hani-app` injects it internally from ~/negmas/hani/
# settings/env.json. The rest of the flags below are forwarded
# verbatim to `panel serve` as Bokeh/panel options.

hani-app --dev --port 5006 --prefix hanapp --index app --allow-websocket-origin study-host.example.org --root-path=/hanapp "$@"

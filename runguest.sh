#!/bin/env bash
# Serve the public, no-auth HANI guest interface on port 5008 (the
# HANI_GUEST_PORT default), mounted at /hanguest behind nginx on
# anac.cs.brown.edu.
#
# `hani-guest` is a thin wrapper that sets HANI_GUEST_MODE=true and then
# execs `panel serve src/hani/app.py --port 5008 <extra args>`, so the
# extra Bokeh/panel flags below behave exactly the same way they do in
# run.sh for the authenticated app.

hani-guest --prefix hanguest --allow-websocket-origin anac.cs.brown.edu --root-path=/hanguest "$@"

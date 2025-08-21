#!/bin/env bash

panel serve  src/hani/register.py --port 5007 --prefix hanreg --allow-websocket-origin anac.cs.brown.edu --root-path=/hanreg --session-token-expiration 900000  $@

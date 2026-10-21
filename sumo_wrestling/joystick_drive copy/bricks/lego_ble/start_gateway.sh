#!/bin/sh
# Install the LEGO SDK once into the App cache, then run the gateway.
# (Not named requirements.txt on purpose: the main container would install it too.)
DEPS=/app/.cache/lego_ble_deps
REQ=/app/bricks/lego_ble/gateway-requirements.txt

if ! cmp -s "$REQ" "$DEPS/installed.txt"; then
  echo "Installing LEGO gateway dependencies..."
  rm -rf "$DEPS"
  pip install --quiet --disable-pip-version-check --target "$DEPS" -r "$REQ" && cp "$REQ" "$DEPS/installed.txt"
fi

export PYTHONPATH="$DEPS"
exec python3 /app/bricks/lego_ble/gateway.py

#!/usr/bin/env bash
# Build a CA bundle that can verify edgar.sci.utah.edu, whose server omits its
# intermediate certificates. The two public intermediates in certs/ are:
#   InCommon Intermediate CA - DVG2C  <- emSign Root TLS CA - G1 (cross-signed)
#   emSign Root TLS CA - G1           <- emSign Root CA - G1 (in the Mozilla trust store)
# Fetched from repository.emsign.com (the leaf's AIA URLs) and checked with `openssl verify`.
set -euo pipefail
cd "$(dirname "$0")/.."

base="${SSL_CERT_FILE:-}"
for c in /root/.ccr/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt; do
  [[ -z "$base" && -f "$c" ]] && base="$c"
done
[[ -n "$base" ]] || base="$(python3 -c 'import certifi; print(certifi.where())')"

cat "$base" certs/edgar-intermediates.pem > .edgar-ca-bundle.pem
echo "wrote .edgar-ca-bundle.pem (base: $base)"

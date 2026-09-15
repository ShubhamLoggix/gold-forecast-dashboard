#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/ShubhamLoggix/gold-forecast-dashboard.git"
APP_DIR="/opt/gold-forecast-dashboard"

if [ "$(id -u)" -ne 0 ]; then
  exec sudo "$0" "$@"
fi

echo "==> Installing Docker Engine + Compose (Ubuntu ARM)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl git openssl iptables-persistent
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker

echo "==> Opening port 80 in the OS firewall (OCI Security List must also allow TCP/80)..."
iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT
netfilter-persistent save

echo "==> Cloning $REPO_URL -> $APP_DIR"
rm -rf "$APP_DIR"
git clone "$REPO_URL" "$APP_DIR"
cd "$APP_DIR/docker"

if [ ! -f .env ]; then
  API_KEY="$(openssl rand -hex 32)"
  printf 'API_KEY=%s\nFRONTEND_PORT=80\n' "$API_KEY" > .env
  chmod 600 .env
fi

echo "==> Building + starting stack (first run downloads ~1GB TimesFM weights; backend start_period is 900s)"
docker compose up -d --build

echo
echo "========================================================="
echo " API key for POST /api/v1/refresh (store it somewhere safe):"
grep '^API_KEY=' .env
echo
docker compose ps
echo
echo "Verify:  curl http://localhost/  and  curl http://localhost/api/v1/health"
echo "Public URL once OCI ingress rule for TCP/80 exists: http://<VM_PUBLIC_IP>/"
echo "========================================================="

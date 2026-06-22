#!/usr/bin/env bash
# Déploiement idempotent de l'app Jonglerie sur un serveur Ubuntu.
# Usage (en root) :  bash deploy/setup.sh
set -euo pipefail
APP_DIR=/opt/jonglerie
REPO=https://github.com/Aschy/Jonglerie.git

echo "==> 1. Paquets système"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3-venv python3-pip nginx ffmpeg >/dev/null

echo "==> 2. Code source dans $APP_DIR"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"
mkdir -p data/jobs models

echo "==> 3. venv runtime + dépendances"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r server/requirements.txt

echo "==> 4. Modèle YOLO (export si absent)"
bash deploy/get_model.sh

echo "==> 5. Service systemd"
cp deploy/jonglerie.service /etc/systemd/system/jonglerie.service
systemctl daemon-reload
systemctl enable --now jonglerie
systemctl restart jonglerie

echo "==> 6. Reverse-proxy nginx"
cp deploy/nginx-jonglerie.conf /etc/nginx/sites-available/jonglerie
ln -sf /etc/nginx/sites-available/jonglerie /etc/nginx/sites-enabled/jonglerie
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

echo "==> 7. Pare-feu"
ufw allow 80/tcp || true

echo "==> ✓ Déployé. Ouvre http://$(curl -s ifconfig.me || echo SERVER_IP)/"

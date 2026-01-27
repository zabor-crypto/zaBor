# Deployment

## Local run (recommended for first setup)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
# export env vars (see docs/ENV_EXAMPLE.md)
python3 src/killswitch.py --config config.yaml
```

## Run in background (screen)
```bash
screen -S killswitch
source .venv/bin/activate
python3 src/killswitch.py --config config.yaml
# Ctrl+A then D
```

## Systemd service (Linux server)
Create `/etc/systemd/system/killswitch.service`:

```ini
[Unit]
Description=Crypto Portfolio Kill-Switch
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/your-repo/killswitch
EnvironmentFile=/opt/your-repo/killswitch/.killswitch_env
ExecStart=/opt/your-repo/killswitch/.venv/bin/python /opt/your-repo/killswitch/src/killswitch.py --config /opt/your-repo/killswitch/config.yaml
Restart=always
RestartSec=5
User=ubuntu
Group=ubuntu

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now killswitch
sudo journalctl -u killswitch -f
```

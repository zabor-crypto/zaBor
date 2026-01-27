# Crypto Portfolio Kill-Switch (v2.0)

Автоматический risk-kill для криптопортфеля: **Binance / Bybit / Bitget**, **Spot + Futures**.

## Что делает
- Периодически снимает equity по каждому `scope` (биржа × тип счета).
- Считает просадку по нескольким окнам.
- При превышении порогов запускает действия Tier A / Tier B:
  - Futures: закрытие лонгов / закрытие всех позиций (лонги+шорты)
  - Spot: продажа blacklisted активов или продажа всего non-stable

Требования: Python 3.8+.

## Быстрый старт (DRY RUN)
```bash
cd killswitch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
# заполните env vars:
#   see docs/ENV_EXAMPLE.md
source ./.killswitch_env

python3 src/killswitch.py --config config.yaml
```

## Режимы запуска
- **Prod:** `python3 src/killswitch.py --config config.yaml`
- **Mock test:** `python3 src/killswitch.py --test-mock`
- **Backtest:** `python3 src/killswitch.py --backtest equity.csv --config config.yaml`

## Документация
- `docs/USER_GUIDE_RU.md` — пользовательское руководство (RU)
- `docs/SECURITY.md` — безопасность и доступы
- `docs/DEPLOYMENT.md` — деплой (screen/systemd)
- `docs/ENV_EXAMPLE.md` — пример env vars

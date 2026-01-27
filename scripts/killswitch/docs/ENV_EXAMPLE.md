# Environment variables

Create a local file (never commit it), for example: `.killswitch_env`

```bash
export BINANCE_API_KEY="..."
export BINANCE_API_SECRET="..."

export BYBIT_API_KEY="..."
export BYBIT_API_SECRET="..."

export BITGET_API_KEY="..."
export BITGET_API_SECRET="..."
export BITGET_PASSWORD="..."   # required by Bitget
```

Load it before running:
```bash
source ./.killswitch_env
```

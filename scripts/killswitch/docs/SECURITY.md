# Security

## Never commit secrets
- Do **not** commit `config.yaml`, `.env`, `.killswitch_env`, API keys, passphrases, or any private endpoints.
- Use environment variables for exchange credentials.

## Minimum API permissions
Grant only:
- Read (balances/positions)
- Trade (place/cancel orders)

Do **not** grant:
- Withdraw
- Transfer
- Sub-account management

## IP whitelist
If your exchange supports IP restrictions, whitelist only your server IP.

## Operational safety checklist (before LIVE)
- `dry_run: true` has been running 24–48h with clean logs
- You enabled only one scope (one exchange + one account type)
- You verified that equity calculation is correct (not 0 / NaN)
- You understand Tier A vs Tier B actions and cooldowns

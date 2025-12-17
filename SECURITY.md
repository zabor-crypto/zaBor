# Security Policy

## Reporting vulnerabilities
Do not open public issues for vulnerabilities or leaked credentials.
Use GitHub Security Advisories for private reporting.

## Rules
- Never commit API keys, secrets, or private endpoints
- `.env` must remain untracked
- Do not log secrets or personally identifying data
- Prefer least-privilege API keys (trade-only vs withdraw)
- Use IP whitelisting if supported by the exchange

## Supported versions
Only the latest release is supported for security fixes.

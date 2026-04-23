# pi-nous

Nous Portal provider extension for pi.

## What it does

- Registers a `nous` provider for pi
- Uses Nous Portal OAuth device flow
- Refreshes OAuth tokens
- Mints short-lived Nous inference agent keys
- Uses the Nous inference API at `https://inference-api.nousresearch.com/v1`

## Files

- `index.ts` — extension entrypoint
- `package.json` — pi package metadata

## Use locally

From this directory:

```bash
pi -e ./index.ts
```

Or add the directory/package in pi config.

## Install as local package

You can also reference it from your pi settings as a local extension/package path.

## Notes

- Based on the working extension developed in `~/.pi/agent/extensions/nous-provider/`
- Uses Hermes-like endpoints:
  - `/api/oauth/device/code`
  - `/api/oauth/token`
  - `/api/oauth/agent-key`
- Includes a manual fallback for Vercel browser checkpoint situations

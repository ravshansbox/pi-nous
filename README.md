# pi-nous

Nous Portal provider extension for pi.

## Install

```bash
pi install git:github.com/ravshansbox/pi-nous
```

## Usage

Pi loads the provider from `./index.ts` and registers a `nous` provider backed by the Nous Portal OAuth flow and inference API.

For example, choose a Nous model in pi, sign in through the device-flow prompt, and the extension will mint an agent key before sending requests to `https://inference-api.nousresearch.com/v1`.

## Development

```bash
npm install
npm run typecheck
```

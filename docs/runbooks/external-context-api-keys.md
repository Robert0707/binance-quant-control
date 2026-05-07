# External Context API Keys

Purpose: connect optional market context providers for Hermes AI Trader. These
feeds are veto/filter inputs only. They do not approve trades by themselves.
Default policy is free-first: use CoinMarketCap if you have the free key, use
public DexScreener, and keep paid providers disabled until you actually want
them.

## Free-First Env

Only this one is needed for the default setup:

```bash
COINMARKETCAP_API_KEY=replace_me
```

Do not paste keys into chat. Keep them local in `.env`.

DexScreener uses a public endpoint and needs no key.

## Paid Optional Env

Leave these blank unless you decide to pay for them:

```bash
CRYPTOPANIC_API_KEY=
GLASSNODE_API_KEY=
ARKHAM_API_KEY=
```

## Registration

CoinMarketCap:
- Register: https://pro.coinmarketcap.com/signup/
- Docs: https://coinmarketcap.com/api/documentation/guides/authentication
- Auth used by this repo: `X-CMC_PRO_API_KEY` header.
- Role: BTC/ETH dominance and global market capital-flow filter.

CryptoPanic:
- Register/docs: https://cryptopanic.com/developers/api/
- Auth used by this repo: `auth_token` query parameter.
- OAuth note: CryptoPanic does not expose an OAuth authorization-code flow for
  this API. Log in on CryptoPanic, copy the API auth token, and store it as
  `CRYPTOPANIC_API_KEY` locally.
- Cost note: paid or limited optional. The default config disables it.
- Role: crypto news and event-risk filter.

Glassnode:
- Register: https://studio.glassnode.com/
- Docs: https://docs.glassnode.com/basic-api/api-key
- Auth used by this repo: `X-Api-Key` header.
- Cost note: paid optional. The default config disables it.
- Role: BTC/ETH on-chain macro metrics filter.

Arkham:
- Register: https://intel.arkm.com/api
- Docs: https://api-guide.intel.arkm.com/
- Auth used by this repo: `API-Key` header.
- Cost note: paid optional. The default config disables it.
- Role: wallet/entity flow and whale-context filter.

## Verify

Check key presence without printing secrets:

```bash
.venv/bin/binance-quant-control external-context-key-status --compact
```

Run a live provider smoke check:

```bash
.venv/bin/binance-quant-control external-context \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,DOGEUSDT,TRXUSDT \
  --compact
```

Healthy free-first result: `coinmarketcap` should be configured after you paste
the key, `dexscreener` should be enabled without a key, and paid providers can
stay in `optional_missing`. Missing or rate-limited providers degrade to neutral
filters instead of creating trade entries.

# Deploying ragsec-scanner

The demo is one Docker container: the FastAPI app, the demo page at `/`, and
the synthetic demo corpus. It is built from the `Dockerfile` in this
repository. **No API key is in the image or the repository**; keys are
supplied as environment variables when the container runs.

Main path: **Oracle Cloud (Always Free Ampere VM) + Coolify**. Render and
Fly.io work too (see the end).

Steps marked **👤 Account owner** need the person who owns the cloud account,
the GitHub repository, or the API key.

## At a glance

| Item | Value |
|---|---|
| Build source | `https://github.com/Koushik-Saha/RAGSec-Scanner`, branch `main`, `Dockerfile` at the repo root |
| Container port | `8000` |
| Health check | `GET /health` → `200 {"status":"ok"}` (the `Dockerfile` also defines a `HEALTHCHECK`) |
| Memory | Use **2 GB or more**. Measured locally (arm64): about 560 MiB after startup, ask, and a short scan. |
| Architecture | Builds and runs on **arm64** (verified locally) and x86_64 |
| Startup | Indexes the demo corpus on boot; no persistent volume needed |

### Environment variables (names only — never commit values)

| Variable | Required? | Value |
|---|---|---|
| `RAGSEC_PROVIDER` | Yes for Groq | `groq` (others: `anthropic` — the default, `openrouter`, `ollama`) |
| `RAGSEC_MODEL` | Yes | Model name for that provider, e.g. `openai/gpt-oss-120b` on Groq |
| `GROQ_API_KEY` | Yes for Groq | **Secret.** Your Groq API key |
| `ANTHROPIC_API_KEY` | Optional | **Secret.** Only if `RAGSEC_PROVIDER=anthropic` |
| `OPENROUTER_API_KEY` | Optional | **Secret.** Only if `RAGSEC_PROVIDER=openrouter` |

Only the selected provider's key is needed. Set keys as **runtime** variables,
never as build arguments, so they never end up in an image layer. The
`RAGSEC_DATA_DIR`, `RAGSEC_CHROMA_DIR`, and `RAGSEC_REPORTS_DIR` defaults are
already set inside the image; leave them alone.

Before deploying, **👤** confirm the model is available to your key (Groq
console → Models). An unavailable model makes every request fail with a 404.

---

## Groq free tier: the demo's ceiling

The deployed app has **no login**. Anyone with the URL can call `/ask` and
`/scan`, and every call spends your Groq key's free-tier budget.

Limits reported by Groq's rate-limit headers for `openai/gpt-oss-120b` on a
free key (checked 2026-11-14; Groq can change them, see
<https://console.groq.com/settings/limits>):

| Limit | Value | Header |
|---|---|---|
| Requests per day | 1,000 | `x-ratelimit-limit-requests` |
| Tokens per minute | 8,000 | `x-ratelimit-limit-tokens` |

What that means for the demo:

- One `/ask` = 1 model call. One full `/scan` (all 10 probes) = 11 model
  calls. So **at most about 90 full scans per day**, fewer when people also
  use `/ask`.
- **Pacing is built in.** The app shares one model client across all
  visitors and spaces Groq requests at least 2.1 s apart (under 30 requests
  per minute). On HTTP 429 it waits for Groq's `retry-after` (or backs off
  2 s, doubling, up to 60 s) and retries up to 6 times. Busy periods get
  slower, not more expensive.
- **When the daily budget is spent**, the app stops calling Groq until the
  `retry-after` time (or 15 minutes) has passed. `/ask` returns
  `503 model_rate_limited`, and scan probes are recorded as ERROR.
- **Pacing limits the rate, not the total.** At the paced maximum, someone
  sending requests nonstop would use up the 1,000 requests/day in about 35
  minutes. To protect the budget:
  1. **👤** Use a **separate Groq API key** just for this demo, so you can
     revoke it without affecting anything else.
  2. Share the URL only with people you trust, and stop the app when the
     demo is over.
  3. For a paid provider (`anthropic`, `openrouter`), **👤** set a spend
     limit in that provider's console first.
- A full `/scan` takes about a minute. If a proxy times out, run probes in
  smaller groups: `POST /scan` with
  `{"probe_ids": ["indirect_injection_body", "injection_via_title"]}`.

---

## Option A: Oracle Cloud + Coolify

### 1. Create the VM (👤 Oracle account owner)

1. Sign in to the Oracle Cloud console. Go to **Compute → Instances →
   Create instance**.
2. **Image:** Canonical Ubuntu (22.04 or 24.04).
   **Shape:** `VM.Standard.A1.Flex` (Ampere, arm64, Always Free eligible).
   Give it at least **2 OCPUs and 12 GB memory**. Coolify needs 2 CPUs and
   2 GB+ on its own, and the first build installs PyTorch.
3. **Boot volume:** 50 GB or more (Docker images and build cache).
4. Add your SSH public key, then create the instance. Note its public IP.

### 2. Open the network ports

Oracle blocks traffic in two places; open both.

1. **VCN security list:** Instance → **Subnet** → **Security Lists** →
   default list → **Add Ingress Rules**, source `0.0.0.0/0`, TCP, destination
   ports `80`, `443`, and `8000`. Port 8000 is Coolify's own dashboard; you
   can remove it later once the dashboard has a domain.
2. **The VM's own firewall:** Oracle's Ubuntu images ship iptables rules that
   allow only SSH. On the VM:
   ```bash
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
   sudo netfilter-persistent save
   ```

### 3. Install Coolify

```bash
ssh ubuntu@<vm-public-ip>
curl -fsSL https://cdn.coollabs.io/coolify/install.sh | sudo bash
```

When it finishes, open `http://<vm-public-ip>:8000` and **👤** create the
admin account straight away (the first visitor to that page becomes the
admin).

### 4. Add the app

Coolify's labels below are from Coolify v4 and may differ slightly between
versions.

1. **Projects → + Add** a project (e.g. `ragsec-demo`), open its environment,
   then **+ New → Resource → Public Repository**.
2. **Repository URL:** `https://github.com/Koushik-Saha/RAGSec-Scanner`,
   **Branch:** `main`.
3. **Build Pack:** `Dockerfile` (Dockerfile location `/Dockerfile`, base
   directory `/`).
4. **Ports Exposes:** `8000`. Leave **Ports Mappings** empty. Coolify's proxy
   routes your domain to the container, and host port 8000 is already used by
   Coolify's dashboard.
5. **Domains:** your domain (e.g. `https://ragsec.example.org`, with a DNS A
   record pointing at the VM) or the generated `sslip.io` address.
6. **Environment Variables:** add the variables from the table above, e.g.
   `RAGSEC_PROVIDER` = `groq`, `RAGSEC_MODEL` = `openai/gpt-oss-120b`,
   `GROQ_API_KEY` = **👤** your key. Leave **Build Variable** *unchecked* for
   all of them; they are runtime-only.
7. **Health Check:** enable it with path `/health` and port `8000`.
8. **Deploy.** The first build takes several minutes (PyTorch and the
   embedding model). The deployment log should end with the container
   becoming healthy and uvicorn listening on `0.0.0.0:8000`.

**Updating:** click **Redeploy** after pushing to `main` (or enable automatic
deploys through Coolify's GitHub App integration). **Changing the key:** edit
`GROQ_API_KEY` under Environment Variables, then **Restart** (no rebuild
needed). **Stopping:** **Stop** on the resource, or delete it.

---

## Post-deploy smoke test

Replace `BASE` with your URL. The scan runs two probes: `context_bleed` makes
no model call and `planted_secret_extraction` makes one, so the test costs one
`/ask` call and one scan call.

```bash
BASE=https://ragsec.example.org

curl -fsS "$BASE/health"
# {"status":"ok"}

curl -fsS -X POST "$BASE/ask" -H 'content-type: application/json' \
  -d '{"question":"What is the daily meal allowance when travelling?"}'
# JSON whose "text" gives the $60/day meal cap, plus "retrieved_chunk_ids"

curl -fsS -X POST "$BASE/scan" -H 'content-type: application/json' \
  -d '{"probe_ids":["planted_secret_extraction","context_bleed"]}'
# JSON with "total": 2, "model": your RAGSEC_MODEL, and no "error" on either probe

curl -fsS -o /dev/null -w '%{http_code}\n' "$BASE/report/html"   # 200
curl -fsS -o /dev/null -w '%{http_code}\n' "$BASE/"              # 200 (demo page)
```

If `/ask` returns `503`, check the troubleshooting table below.

Known quirk with `openai/gpt-oss-120b`: it often cites sources as
`【chunk-id】` instead of `[chunk-id]`, which the citation parser does not
read. Those answers show `"citations": []` and `"grounded": false` even when
the answer is right.

---

## Other hosts

Both build the same `Dockerfile` and need the same environment variables
(see the table above).

### Render

1. **👤** Sign in at <https://dashboard.render.com>. Click **New + → Web
   Service** and connect the GitHub repository `Koushik-Saha/RAGSec-Scanner`
   (branch `main`). **Language:** Docker, **Dockerfile Path:**
   `./Dockerfile`.
2. **Instance Type:** 2 GB RAM (smaller instances run out of memory).
3. **Environment Variables:** the provider variables from the table above,
   plus `PORT` = `8000` (tells Render which port the container listens on).
   **👤** Only the key owner should paste the key.
4. **Advanced → Health Check Path:** `/health`. Then **Deploy Web Service**.
5. Run the smoke test above against `https://<name>.onrender.com`.

### Fly.io

1. **👤** Create an account at <https://fly.io>, install `flyctl`, and run
   `fly auth login`.
2. From the repository folder:
   `fly launch --no-deploy --name ragsec-scanner-demo --region sjc` (answer
   **No** to databases).
3. Replace `fly.toml` with (keep your own `app` name and region):
   ```toml
   app = "ragsec-scanner-demo"
   primary_region = "sjc"

   [build]
     dockerfile = "Dockerfile"

   [env]
     RAGSEC_PROVIDER = "groq"
     RAGSEC_MODEL = "openai/gpt-oss-120b"

   [http_service]
     internal_port = 8000
     force_https = true
     auto_stop_machines = "stop"
     auto_start_machines = true
     min_machines_running = 0

     [[http_service.checks]]
       grace_period = "30s"
       interval = "30s"
       method = "GET"
       timeout = "5s"
       path = "/health"

   [[vm]]
     memory = "2gb"
     cpu_kind = "shared"
     cpus = 1
   ```
   `fly.toml` contains no secrets.
4. **👤** Store the key as a Fly secret (encrypted, never in `fly.toml`):
   `fly secrets set GROQ_API_KEY=<your-groq-key>`.
5. `fly deploy`, then run the smoke test against `https://<app>.fly.dev`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Container restarts; logs show `Killed` or out-of-memory | Too little RAM | Give the VM/instance 2 GB or more |
| Coolify dashboard or the app unreachable on Oracle | Port closed in the security list or the VM's iptables | Open the port in **both** places (Option A, step 2) |
| `/ask` → `503 not_configured`, "RAGSEC_MODEL is not set" | Missing variable | Set `RAGSEC_MODEL` |
| `/ask` → `503 not_configured`, "GROQ_API_KEY is not set …" | Key missing for the selected provider | Set the key named in the message (or fix `RAGSEC_PROVIDER`) |
| `/ask` → `503 not_configured`, "RAGSEC_PROVIDER … is not supported" | Typo in the provider | Use `anthropic`, `groq`, `openrouter`, or `ollama` |
| `/ask` → `503`, "rejected the server's credentials" | Wrong or revoked key | **👤** Create a new key and update the variable |
| `/ask` → `502 model_api_error`, "The model API call failed (NotFoundError)" | Model not available to the key | Pick a model listed for your key |
| `/ask` → `503 model_rate_limited` | Groq per-minute or daily limit | Wait; see "Groq free tier" above |
| `/scan` times out | Proxy timeout on a long request | Run fewer probes per request |
| Render: "No open ports detected" | Port not set | Set `PORT=8000` |

## After deploying

**👤** Put the live URL at the top of `README.md` and commit it.

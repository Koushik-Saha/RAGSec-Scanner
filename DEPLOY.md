# Deploying ragsec-scanner

Two options: **Render** (all in the browser) or **Fly.io** (command line).
Both build the `Dockerfile` in this repository and need the same two
environment variables:

| Variable | Value | Secret? |
|---|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (`sk-ant-...`) | **Yes** — never commit it |
| `RAGSEC_MODEL` | The Claude model to test, e.g. `claude-opus-5-5` | No |

Steps marked **👤 Account owner** need the person who owns the Render or
Fly.io account, the GitHub repository, or the Anthropic API key. Everything
else can be done by anyone following along.

## Before you start

- **Memory: use an instance with 2 GB of RAM.** Measured locally, the
  container uses about 830 MiB after startup and about 880 MiB after handling
  requests (PyTorch and the embedding model). Instances with 512 MB, including
  most free tiers, will run out of memory. 1 GB may work but leaves little
  headroom during a scan.
- **Startup:** about 11 seconds locally (the image includes the embedding
  model; on boot it indexes the demo corpus). The index lives inside the
  container and is rebuilt on every start, so no persistent disk is needed.
- **Long requests:** a full `/scan` makes about 13 model calls and can take a
  minute or more. If the platform's proxy times out, run probes in smaller
  groups: `POST /scan` with `{"probe_ids": ["indirect_injection_body", "injection_via_title"]}`.
- **👤 Cost and access — read this.** The deployed app has **no login**.
  Anyone who finds the URL can call `/ask` and `/scan`, and every call is
  billed to your Anthropic key. Before deploying:
  1. **👤** In the Anthropic Console, create a **separate API key** just for
     this deployment, so you can revoke it without affecting anything else.
  2. **👤** In the Anthropic Console, set a **monthly spend limit** you are
     comfortable losing.
  3. Share the URL only with people you trust, and delete the service when the
     demo is over.

---

## Option A: Render

1. **👤** Sign in at <https://dashboard.render.com> (create an account if
   needed, and add a payment method if your plan requires it; the 2 GB
   instance is a paid instance type).
2. Click **New +** (top right) → **Web Service**.
3. Under **Source Code**, choose **Git Provider** → **GitHub**.
   - **👤** If this is the first time, click **Connect GitHub** and authorize
     Render to read `Koushik-Saha/RAGSec-Scanner`.
   - Select the **RAGSec-Scanner** repository and click **Connect**.
4. Fill in the settings:
   - **Name:** `ragsec-scanner` (becomes `https://ragsec-scanner.onrender.com`
     if available; otherwise Render adds a suffix).
   - **Language / Runtime:** **Docker** (Render detects the `Dockerfile`).
   - **Branch:** `main`
   - **Region:** any; pick the one closest to you.
   - **Root Directory:** leave empty.
   - **Dockerfile Path:** `./Dockerfile` (the default).
5. **Instance Type:** choose one with **2 GB RAM** (see "Before you start").
6. **Environment Variables** → **Add Environment Variable**, three times:
   - `ANTHROPIC_API_KEY` = your key. **👤** Only the key owner should paste
     this.
   - `RAGSEC_MODEL` = `claude-opus-5-5`
   - `PORT` = `8000` — tells Render which port the container listens on.
7. Open **Advanced** and set **Health Check Path** to `/health`.
8. Click **Deploy Web Service** (or **Create Web Service**).
9. Wait for the build (several minutes the first time: it installs PyTorch and
   downloads the embedding model). The log should end with:
   ```
   Uvicorn running on http://0.0.0.0:8000
   ```
   and the status at the top should turn **Live**.
10. Check it (replace the URL with yours):
    ```bash
    curl https://ragsec-scanner.onrender.com/health
    # {"status":"ok"}
    ```
    Then open the URL in a browser to see the demo page, ask a question, and
    run a scan.

**Updating:** every push to `main` redeploys automatically (Render's default
"Auto-Deploy: Yes"). **Changing the key:** **Environment** tab → edit
`ANTHROPIC_API_KEY` → **Save, rebuild, and deploy**.
**Stopping:** **Settings** → bottom of the page → **Suspend Web Service** (or
**Delete Web Service**).

---

## Option B: Fly.io

1. **👤** Create an account at <https://fly.io> and add a payment method
   (Fly.io requires one before launching apps).
2. Install the `flyctl` command-line tool:
   ```bash
   brew install flyctl        # macOS; see https://fly.io/docs/flyctl/install/ for others
   ```
3. **👤** Log in (opens a browser window):
   ```bash
   fly auth login
   ```
4. From the repository folder, create the app without deploying yet. Pick an
   app name that is free on Fly.io (it becomes `https://<name>.fly.dev`) and a
   region code close to you (`fly platform regions` lists them):
   ```bash
   cd RAGSec-Scanner
   fly launch --no-deploy --name ragsec-scanner-demo --region sjc
   ```
   If it asks whether to set up databases or other services, answer **No**.
   It writes a `fly.toml` file.
5. Replace the contents of `fly.toml` with the following (keep your own `app`
   name and region):
   ```toml
   app = "ragsec-scanner-demo"
   primary_region = "sjc"

   [build]
     dockerfile = "Dockerfile"

   [env]
     RAGSEC_MODEL = "claude-opus-5-5"

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
   `fly.toml` contains no secrets and can be committed.
   `auto_stop_machines` stops the machine when idle to save money; the first
   request after a stop waits for it to start again (about 10–20 seconds).
6. **👤** Store the API key as a secret (it is encrypted and never written to
   `fly.toml`):
   ```bash
   fly secrets set ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```
7. Deploy:
   ```bash
   fly deploy
   ```
   The first build takes several minutes. It finishes with a line like
   `Visit your newly deployed app at https://ragsec-scanner-demo.fly.dev/`.
8. Check it:
   ```bash
   fly status                                   # machine should be "started", checks "passing"
   curl https://ragsec-scanner-demo.fly.dev/health
   # {"status":"ok"}
   fly open                                     # opens the demo page in a browser
   ```
9. If something is wrong, read the logs:
   ```bash
   fly logs
   ```

**Updating:** run `fly deploy` again after pulling changes. **Changing the
key:** `fly secrets set ANTHROPIC_API_KEY=...` (restarts the app).
**Stopping:** `fly scale count 0` to stop all machines, or
`fly apps destroy ragsec-scanner-demo` to delete the app.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Service restarts, logs show `Killed` or out-of-memory | Instance too small | Use 2 GB RAM |
| `/ask` returns `503 not_configured` with "RAGSEC_MODEL is not set" | Missing variable | Set `RAGSEC_MODEL` |
| `/ask` returns `503 not_configured` with "ANTHROPIC_API_KEY is not set" | Missing secret | Set `ANTHROPIC_API_KEY` |
| `/ask` returns `503` "rejected the server's credentials" | Wrong or revoked key | **👤** Create a new key and update the secret |
| `/ask` returns `503 model_rate_limited` | Anthropic rate limit | Wait and retry; check the key's limits in the Console |
| `/scan` times out in the browser | Platform proxy timeout on long requests | Run fewer probes per request (see "Before you start") |
| Render: "No open ports detected" | Port not set | Make sure `PORT=8000` is set |

## After deploying

**👤** Put the live URL at the top of `README.md` and commit it:

```bash
git commit -am "docs: live demo URL"
git push
```

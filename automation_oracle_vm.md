# Automation Plan: Oracle OCI VM (`oracle-micro-free`)

## 1. Target Infrastructure
- **Host:** `oci-vm` (`136.248.75.10`, user `opc`, `VM.Standard.E2.1.Micro`, Oracle Linux 9, 945 MiB RAM + 3.0 GiB Swap)
- **Remote Project Path:** `/home/opc/ple-agent-harness`
- **Systemd Unit:** `/etc/systemd/system/ple-autonomous.service`
- **Dashboard Integration:** `/opt/oci-dashboard/server.py` (`AGENTS` list -> `id: "ple"`, `service: "ple-autonomous.service"`, `competition: "physical-engineerings-last-exam"`, `minimize: false`, `daily_limit: 5`)

## 2. Memory & SELinux Safeguards
- **SELinux:** `.env` is loaded directly in Python (`load_dotenv()` in `src/autonomous_orchestrator.py`) instead of `EnvironmentFile=` in systemd.
- **RAM Budget:** `MemoryHigh=120M`, `MemoryMax=180M`, `Nice=10`, explicit `gc.collect()` after every cycle, and pure-stdlib orchestration so the 4 existing autonomous agents on `oci-vm` remain undisturbed.

## 3. Deployment Commands (Non-Interactive)
```bash
rsync -avz --delete \
  --exclude '.git' --exclude 'data/raw' --exclude 'data/ple_repo' --exclude 'data/reference_kernels' \
  ./ oci-vm:/home/opc/ple-agent-harness/
```
Then install and enable `ple-autonomous.service`, register `ple` in `/opt/oci-dashboard/server.py`, restart `oci-dashboard.service`, and verify `/api/status`.

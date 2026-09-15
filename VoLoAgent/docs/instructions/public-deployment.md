<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Running the orchestrator across machines

> **Use case:** the orchestrator runs on one machine; the eval client
> (robolab, LIBERO, real-robot driver, ...) runs on a different machine
> on the same trusted private network. Both machines can reach each other
> directly.

---

## TL;DR — three steps

Say the orchestrator host is `192.0.2.42` and the robot machine is
`198.51.100.10` (RFC 5737 documentation addresses).

1. **Orchestrator host (192.0.2.42)** — launch as usual. The default
   `--host 0.0.0.0` already accepts connections from any interface, so
   no flag changes are needed:

   ```bash
   vlm-orchestrator --port 8001 \
     --vla-host 127.0.0.1 --vla-port 8000 \
     --mode subgoal_scene_edit ...
   ```

   The startup banner prints the reachable addresses. Pick the
   non-loopback one, e.g.:

   ```
   ================================================================
    vlm-orchestrator listening on port 8001
      ws://127.0.0.1:8001    (lo)
      ws://192.0.2.42:8001   (eno1)
      ws://172.17.0.1:8001   (docker0)
    Eval-client flag: --remote-host <one of the above>
   ================================================================
   ```

2. **Robot machine (198.51.100.10)** — point the eval client's
   `--remote-host` at the orchestrator's private-network IP:

   ```bash
   cd /path/to/RoboLab
   python policies/volo/run.py \
     --policy pi05 \
     --remote-host 192.0.2.42 --remote-port 8001 \
     ...
   ```

3. **Orchestrator host firewall** — allow inbound TCP on the
   orchestrator port from your trusted private-network range. On Ubuntu / `ufw`:

   ```bash
   sudo ufw allow from 192.0.2.0/24 to any port 8001 proto tcp
   ```

   Substitute the actual trusted network range used by your environment.

That's it. No code change, no token, no TLS — within a trusted private network
the link is plain `ws://`.

---

## Why this works without orchestrator code changes

There are two different addresses people mix up:

- **Bind address** (server): which interfaces *am I willing to accept
  connections on*?
- **Connect address** (client): which IP *am I dialing*?

A Linux box has multiple interfaces:

- `lo` (loopback, `127.0.0.1`) — reachable only from the same machine.
- `eno1` / `eth0` (the private-network NIC, e.g. `192.0.2.42`) —
  reachable from any other machine on that network.

When a server binds to `127.0.0.1:8001`, the kernel only delivers
packets that arrived over `lo`. A packet arriving on the NVIDIA NIC is
dropped — the other machine literally can't reach it, firewall or no.

When a server binds to `0.0.0.0:8001` (the orchestrator's default,
`cli.py:62`), the kernel delivers packets from *every* interface.
A connection from `198.51.100.10` is accepted just like a loopback
connection.

The VLA policy, grasp server, and HITL UI all live on the *same host*
as the orchestrator, so they stay on `127.0.0.1` and never need to be
exposed. Only the orchestrator's inbound port faces the network.

---

## Troubleshooting

**"Connection refused" from the robot machine.**
- Confirm the orchestrator banner shows `ws://<your-IP>:<port>`. If it
  only shows `127.0.0.1`, you launched with `--host 127.0.0.1`.
- From the orchestrator host, run `ss -tlnp | grep <port>`. You should
  see `0.0.0.0:<port>`.
- From the robot machine, `nc -vz <orch-ip> <port>` to check raw
  reachability.

**Reachability test passes but the WS handshake hangs.**
- Likely a host firewall on the orchestrator. Check
  `sudo ufw status` / `sudo iptables -L -n`.

**`ip addr` doesn't appear in the banner.**
- The banner uses `ip -4 -o addr show`. On the rare host without
  iproute2 it falls back to a single primary-IP probe + loopback. Run
  `ip addr` manually to find the right NIC.

---

## What this deployment does *not* include

- **No authentication.** Anyone on the NVIDIA network who can reach
  the port can drive the policy. Treat the port like any other
  unauthenticated dev service — don't expose it to public internet, and
  don't run untrusted instructions through it.
- **No TLS.** Plain `ws://`. If you ever need encryption, terminate
  TLS in front of the orchestrator with nginx / Caddy rather than
  patching `websockets.serve` here.
- **HITL browser UI** still binds `0.0.0.0:8002` by default. If you
  want a remote operator to use it, prefer an SSH tunnel
  (`ssh -L 8002:localhost:8002 orch-host`) over exposing it directly.

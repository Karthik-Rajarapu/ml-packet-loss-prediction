# ENVIRONMENT_SETUP.md
## Phase 0 — Networking Environment Validation

Status: **Inspection only. Nothing installed, modified, or smoke-tested.** WSL2 itself is not yet enabled on this machine, which blocks every downstream check (Ubuntu, Mininet, iperf3, tc, Open vSwitch) — those are recorded as **NOT TESTED**, not as failures, because there is currently no Linux environment for them to run in.

Legend: **AVAILABLE** — confirmed present/working · **MISSING** — confirmed absent, needs installing · **NOT TESTED** — could not be checked because a prerequisite is missing · **NOT APPLICABLE** — doesn't apply to this environment.

---

## 1. Operating System

| Item | Result | Status |
|---|---|---|
| OS | Windows 10 Pro, version 22H2, build 19045.7725 | AVAILABLE |
| Minimum required for WSL2 | build ≥ 19041 | AVAILABLE (build exceeds requirement) |
| PowerShell | 5.1.19041.7725, Desktop edition (Windows PowerShell, not PS7 Core) | AVAILABLE |
| CPU | Intel Core i5-7300U — 2 physical cores / 4 logical (hyper-threaded) | AVAILABLE, but modest — keep topologies small |
| RAM | 8,073 MB total; only 629 MB free at inspection time | AVAILABLE, but **low headroom** — flagged as a real constraint below |
| Disk (C:) | 35.83 GB free of 116.04 GB | AVAILABLE |
| Hardware virtualization (VT-x, SLAT) | Enabled in firmware per `systeminfo` Hyper-V Requirements | AVAILABLE |

## 2. WSL Status

| Item | Result | Status |
|---|---|---|
| `wsl.exe` binary present | `C:\WINDOWS\system32\wsl.exe` | AVAILABLE |
| Windows feature `Microsoft-Windows-Subsystem-Linux` | `InstallState = 2` (Disabled, not Absent) | **MISSING** (present on this SKU, just switched off) |
| Windows feature `VirtualMachinePlatform` | `InstallState = 2` (Disabled) | **MISSING** |
| `wsl --status` | exit code 50 (WSL not usable until features enabled) | MISSING |
| `wsl -l -v` / `wsl -l -q` | falls through to generic usage help, no distro list returned | MISSING (no distros — expected, since WSL isn't enabled) |
| WSL default version set to 2 | — | NOT TESTED (can't set until WSL is enabled) |
| Hypervisor currently running | `HypervisorPresent = False` | MISSING (expected — VM platform feature is off) |

## 3. Linux Distribution Status

| Item | Result | Status |
|---|---|---|
| Ubuntu installed | No distributions of any kind installed | **MISSING** |
| Ubuntu version | — | NOT TESTED |
| Any other WSL distro present | None found | MISSING |

## 4. Python Status (inside Ubuntu/WSL)

| Item | Result | Status |
|---|---|---|
| Python 3 inside WSL | — | NOT TESTED (no WSL distro exists yet) |
| pip inside WSL | — | NOT TESTED |

Note: Windows-side Python (if any is installed globally) is irrelevant here — Mininet and the experiment scripts must run inside the Linux environment, not on Windows directly.

## 5. Mininet Status

| Item | Result | Status |
|---|---|---|
| Mininet installed | — | NOT TESTED (requires Ubuntu/WSL first) |
| Mininet version | — | NOT TESTED |
| `mn --test pingall` | — | NOT TESTED |

## 6. iperf3 Status

| Item | Result | Status |
|---|---|---|
| iperf3 inside WSL | — | NOT TESTED |
| iperf3 on Windows directly | Not checked/not relevant — traffic generation must happen inside the Linux emulated topology, not on the Windows host | NOT APPLICABLE |

## 7. tc / iproute2 Status

| Item | Result | Status |
|---|---|---|
| `tc` inside WSL | — | NOT TESTED |
| iproute2 package inside WSL | — | NOT TESTED |
| `tc` on Windows | Does not exist — `tc`/netem is a Linux-kernel-only facility, Windows has no equivalent | NOT APPLICABLE |
| `ping` on Windows | Present (native Windows `ping.exe`) | AVAILABLE, but **not sufficient** — RTT/jitter measurements for the experiments must be taken with Linux `ping` inside Mininet's network namespaces, not the Windows host tool |

## 8. Open vSwitch Status

| Item | Result | Status |
|---|---|---|
| Open vSwitch inside WSL | — | NOT TESTED |
| `ovs-vsctl --version` | — | NOT TESTED |

## 9. sudo / Privilege Status

| Item | Result | Status |
|---|---|---|
| `sudo` inside WSL Ubuntu | — | NOT TESTED (no WSL user account exists yet — created on first Ubuntu launch) |
| Root/namespace-creation capability for Mininet | — | NOT TESTED |

## 10. Missing Components (summary)

In dependency order — each blocks the next:

1. Windows feature `Microsoft-Windows-Subsystem-Linux` — disabled
2. Windows feature `VirtualMachinePlatform` — disabled
3. WSL2 set as default version — blocked by (1)/(2)
4. Ubuntu distribution — blocked by (1)–(3)
5. Mininet, Open vSwitch, iperf3, iproute2, Python3/pip — blocked by (4), installed via `apt` once Ubuntu exists

Nothing below step 2 (Windows features) can be verified until step 2 is resolved — this is why so much of this report reads NOT TESTED rather than PASS/FAIL: there is genuinely no environment yet to run a test against.

## 11. Installation Requirements

None of the following has been executed. Each entry states what it does, why it's needed, what it touches, and its privilege/risk profile, so you can approve them individually or as a batch.

| # | Command | What it installs | Why needed | Touches | Admin required | Risk/limitations |
|---|---|---|---|---|---|---|
| 1 | `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart` | Windows WSL base feature | Prerequisite for any WSL usage | Windows (OS feature flag) | **Yes** | Low risk, fully reversible (can disable again); requires reboot to take effect |
| 2 | `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart` | Windows VM platform feature | Backs the lightweight VM WSL2 needs | Windows (OS feature flag) | **Yes** | Same as above |
| — | *(reboot)* | — | Required for (1)/(2) to take effect | Windows | N/A (user action) | Standard restart; close other work first |
| 3 | `wsl --set-default-version 2` | Sets WSL2 as default for new distros | WSL1 lacks real network namespaces/veth — Mininet **requires** WSL2 | WSL config only | No (user-level) | None significant |
| 4 | `wsl --install -d Ubuntu` | Downloads and installs Ubuntu (LTS) as a WSL2 distro | Provides the actual Linux kernel/userspace Mininet needs | Downloads ~600MB–1GB; creates a WSL VM disk (grows as used, several GB) | No (user-level, but needs features 1–2 enabled first) | First launch prompts for a UNIX username/password (local to WSL only, unrelated to Windows account) |
| 5 | `sudo apt update && sudo apt install -y mininet iperf3 iproute2 iputils-ping python3 python3-pip` | Mininet, Open vSwitch (pulled in as a dependency), iperf3, iproute2 (`tc`/`ss`), ping, Python3, pip | The actual networking/emulation toolchain the project depends on | Inside the Ubuntu WSL filesystem only — does not touch Windows | `sudo` inside WSL (not Windows admin) | Standard apt install; ~200–400MB of packages; no Windows-side effect |
| 6 | `sudo mn --test pingall` (verification, not installation) | — | Confirms Mininet can create namespaces/veth and hosts can reach each other | WSL only | `sudo` inside WSL | None — read/verify only |

**Nothing here modifies global Windows networking settings or the firewall.** Step 1–2 are the only ones that touch Windows itself (as OS feature toggles, not network config), and both require a reboot — flagging clearly per the safety rules above.

## 12. Known Windows/WSL2 Considerations

- **WSL1 vs WSL2**: WSL1 does not implement real network namespaces/veth pairs, so Mininet cannot run on it. WSL2 (a real Linux kernel in a lightweight VM) is mandatory — step 3 above is not optional.
- **Low current free RAM (629 MB of 8 GB)**: this is the most concrete risk on this machine. WSL2 by default can claim up to 50% of host RAM. Recommend creating a `.wslconfig` (under `%UserProfile%`) capping WSL2 memory (e.g. `memory=3GB`) once installed, and closing other applications before running experiments. This affects experiment scale (keep early topologies to 2–4 hosts, short durations), not feasibility.
- **2 physical cores**: fine for a small dumbbell topology and short `iperf3` runs; large parallel parameter sweeps should be run sequentially rather than concurrently on this machine.
- **WSL2 NAT networking**: WSL2 sits behind a virtual NAT adapter; irrelevant to Mininet itself (which builds its own topology *inside* the WSL2 VM), but will matter later if the Phase-8 dashboard needs to be reached from a Windows browser — not a Phase 0 concern.
- **No blockers found in firmware/hardware** — virtualization is enabled in BIOS already, so no BIOS-level change is anticipated (unlike some machines where VT-x must be manually enabled in firmware first).

---

## Environment Verdict

**READY WITH FIXES**

The hardware and OS fully qualify for WSL2 (firmware virtualization enabled, build ≥19041, adequate disk). Nothing found is a hard blocker. However, WSL2 itself is not yet enabled, no Linux distribution exists, and therefore none of Mininet/iperf3/tc/Open vSwitch could be tested — this is an environment that needs a defined, low-risk setup sequence (Section 11, items 1–5) before Phase 1 can begin. The one genuine risk factor to actively manage (not just install around) is the low current free-RAM headroom, which should shape how large early experiments are, not whether the approach is viable at all.

### Recommended Next Step (READY WITH FIXES)

Before Phase 1 can start, in order:

1. **You (or I, with your explicit go-ahead) run the two elevated DISM commands** in Section 11 (#1–#2) to enable the WSL and VirtualMachinePlatform features.
2. **You reboot the machine** — this step needs to be done by you regardless, since I can't trigger a Windows restart from here.
3. After reboot, run `wsl --set-default-version 2` and `wsl --install -d Ubuntu` (Section 11, #3–#4), and complete the first-launch UNIX username/password prompt.
4. Install the toolchain inside Ubuntu (`sudo apt install mininet iperf3 iproute2 iputils-ping python3 python3-pip`, Section 11 #5).
5. Only then re-run this Phase 0 inspection to fill in the currently NOT TESTED rows and execute the Step 4 smoke test (`mn --test pingall`, bandwidth-limited link, delay, and the UDP-over-limited-bandwidth congestion demonstration) — that smoke test is still pending and was not run this session because its prerequisites don't exist yet.

Once the smoke test in Section 5 of the original plan passes end-to-end (Mininet starts, two hosts ping, iperf3 moves traffic, `tc` shapes bandwidth/delay, and controlled congestion/loss is observed), Phase 1 (single dumbbell topology script + one manual experiment run, per PROJECT_PLAN.md Phase 1) can begin.

---

*This file will be updated with real command output and PASS/FAIL results once WSL2/Ubuntu/Mininet are actually installed and Step 4's smoke test can run.*

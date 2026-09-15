r"""Minimal dumbbell topology for the packet-loss experiments.

    h1 --\                                  /-- h3
          s1 ====[ shaped bottleneck ]==== s2
    h2 --/                                  \-- h4

Left-side hosts (h1, h2, ...) connect to s1 over fast, unshaped links.
Right-side hosts (h3, h4, ...) connect to s2 the same way. The ONLY
shaped link is s1<->s2, which is where bandwidth/delay/queue limits are
applied -- this is what makes it a genuine bottleneck rather than just
a set of independently-throttled access links, matching the standard
dumbbell topology used to demonstrate congestion in networking courses
(PROJECT_PLAN.md Section 6).

This module requires Mininet's own Python package, which is installed
via `apt install mininet` inside a Linux (WSL2 Ubuntu) environment --
it cannot run on Windows directly. The import is deferred into
build_dumbbell_net() so the rest of this package (schema, collectors,
targets, config) stays importable and unit-testable on any machine.
"""

from __future__ import annotations

from network.config import ExperimentConfig


class MininetUnavailableError(RuntimeError):
    """Raised when Mininet's Python API cannot be imported."""


def _import_mininet():
    try:
        from mininet.link import TCLink  # noqa: F401
        from mininet.net import Mininet  # noqa: F401
        from mininet.topo import Topo  # noqa: F401
    except ImportError as exc:
        raise MininetUnavailableError(
            "Mininet's Python package is not importable in this environment. "
            "It must be installed inside WSL2 Ubuntu via `sudo apt install mininet` "
            "and this script run with `sudo python3` inside that Ubuntu shell -- "
            "see docs/ENVIRONMENT_SETUP.md. It cannot run directly on Windows."
        ) from exc
    return TCLink, Mininet, Topo


def build_dumbbell_topo_class(config: ExperimentConfig):
    """Return a Mininet Topo subclass wired according to `config`.

    Built dynamically (rather than a single hardcoded class) so
    n_left_hosts / n_right_hosts / bottleneck parameters are genuinely
    configurable per PROJECT_PLAN's "reusable, not hardcoded" requirement.
    """
    _, _, Topo = _import_mininet()

    class DumbbellTopo(Topo):
        def build(self):
            s1 = self.addSwitch("s1")
            s2 = self.addSwitch("s2")

            for i in range(1, config.n_left_hosts + 1):
                host = self.addHost(f"h{i}")
                # Fast, unshaped access link -- the bottleneck is s1<->s2 only.
                self.addLink(host, s1, bw=1000)

            right_start = config.n_left_hosts + 1
            for i in range(right_start, right_start + config.n_right_hosts):
                host = self.addHost(f"h{i}")
                self.addLink(host, s2, bw=1000)

            self.addLink(
                s1, s2,
                bw=config.bottleneck_bw_mbps,
                delay=f"{config.bottleneck_delay_ms}ms",
                max_queue_size=config.bottleneck_queue_pkts,
                use_htb=True,
            )

    return DumbbellTopo


def build_dumbbell_net(config: ExperimentConfig):
    """Build and start a Mininet network for `config`. Caller must net.stop() it.

    Uses Mininet's bundled default switch/controller (same defaults the
    `mn` CLI itself uses for `sudo mn --test pingall`), so a topology that
    works here should behave the same as the Phase 0 smoke tests.
    """
    TCLink, Mininet, _ = _import_mininet()
    topo_cls = build_dumbbell_topo_class(config)
    net = Mininet(topo=topo_cls(), link=TCLink, autoSetMacs=True)
    net.start()
    return net

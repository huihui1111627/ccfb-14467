"""基础自洽性测试:python3 -m test_meshnet(或直接 python3 test_meshnet.py)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meshnet.model import Network, SINR_THRESHOLD_DB, default_channels, overlap_factor
from meshnet.planner import plan_network, build_conflict_graph
from meshnet.simulate import simulate
from meshnet.baselines import exhaustive_optimal
from run_demo import build_scenic_area, build_small


def test_channel_overlap():
    chs = {c.idx: c for c in default_channels()}
    assert overlap_factor(chs[1], chs[1]) == 1.0
    assert overlap_factor(chs[1], chs[6]) == 0.0       # 完全不重叠
    assert 0 < overlap_factor(chs[1], chs[3]) < 1      # 部分重叠按比例折算
    assert overlap_factor(chs[1], chs[36]) == 0.0      # 跨频段不干扰


def test_plan_consistency():
    net = build_scenic_area()
    plan = plan_network(net)
    # 每条链路两端必须有射频调到同一信道
    for lid, ci in plan.link_channel.items():
        lk = net.links[lid]
        assert ci in plan.radio_channels[lk.a], f"link {lid} 端点 {lk.a} 无信道 {ci}"
        assert ci in plan.radio_channels[lk.b], f"link 端点 {lk.b} 无信道 {ci}"
    # 射频数不超限
    for nid, radios in plan.radio_channels.items():
        assert len(radios) <= net.nodes[nid].radios
    # 承载链路都分到了时隙
    used = set()
    for path in plan.routes.values():
        for u, v in zip(path, path[1:]):
            used.add(next(l for l in net.adj[u] if net.links[l].other(u) == v))
    for lid in used:
        assert plan.link_slots.get(lid), f"承载链路 {lid} 没分到时隙"


def test_slot_sinr_feasible():
    """规划出的每个时隙,在累加干扰下所有接收都必须过门限."""
    net = build_scenic_area()
    plan = plan_network(net)
    chans = {c.idx: c for c in plan.channels}
    for s, lids in enumerate(plan.slot_links):
        active = [(net.links[l].a, net.links[l].b, plan.link_channel[l]) for l in lids]
        for tx, rx, ci in active:
            interf = [(t2, chans[c2]) for t2, r2, c2 in active if t2 != tx]
            sinr = net.sinr_db(rx, tx, chans[ci], interf)
            assert sinr >= SINR_THRESHOLD_DB, f"时隙 {s} 链路 {tx}->{rx} SINR {sinr:.1f} 不过"


def test_sim_beats_baselines():
    from meshnet.baselines import baseline_single_channel, baseline_alternate
    net = build_scenic_area()
    s_main = simulate(net, plan_network(net))
    s_single = simulate(net, baseline_single_channel(net))
    s_alt = simulate(net, baseline_alternate(net))
    assert s_main.worst_throughput > s_single.worst_throughput
    assert s_main.worst_throughput > s_alt.worst_throughput


def test_dfs_no_outage():
    net = build_scenic_area()
    plan = plan_network(net)
    sim = simulate(net, plan, radar={5: [52, 56]})
    assert sim.dfs_event["outage_slots"] == 0


def test_exhaustive_gap_small():
    net = build_small()
    s_ours = simulate(net, plan_network(net))
    p_opt, _ = exhaustive_optimal(net)
    s_opt = simulate(net, p_opt)
    assert s_ours.worst_throughput >= 0.9 * s_opt.worst_throughput


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} 个测试全部通过")

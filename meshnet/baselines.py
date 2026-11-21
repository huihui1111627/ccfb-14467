"""对照方案:全网同一信道、按经验隔跳换信道、小规模穷举最优."""
from __future__ import annotations

import itertools
import math

from .model import Network, default_channels
from .planner import (FRAME_SLOTS, Plan, assign_channels, compute_routes,
                      link_loads, plan_dfs_backup, schedule_slots)


def _finish_plan(net, radio, link_ch, routes, loads, frame_slots):
    slot_links, link_slots = schedule_slots(net, link_ch, loads,
                                            default_channels(), radio, frame_slots)
    p = Plan(net=net, radio_channels=radio, link_channel=link_ch,
             routes=routes, slot_links=slot_links, link_slots=link_slots,
             channels=default_channels())
    p.backup_channels, p.link_backup = plan_dfs_backup(net, p)
    return p


def baseline_single_channel(net: Network, ch_idx: int = 6,
                            frame_slots: int = FRAME_SLOTS) -> Plan:
    """所有射频同一信道 —— 冲突图最密,全靠分时."""
    routes = compute_routes(net)
    loads = link_loads(net, routes)
    radio = {n.id: [ch_idx] * n.radios for n in net.nodes.values()}
    link_ch = {lid: ch_idx for lid in net.links}
    return _finish_plan(net, radio, link_ch, routes, loads, frame_slots)


def baseline_alternate(net: Network, ch_idxs=(1, 6, 11),
                       frame_slots: int = FRAME_SLOTS) -> Plan:
    """装机师傅经验做法:沿路由隔跳换信道(1/6/11 轮换),不考虑地形和负载."""
    routes = compute_routes(net)
    loads = link_loads(net, routes)
    # 节点深度 -> 信道
    from .simulate import _depths
    depth = _depths(net)
    radio = {}
    for n in net.nodes.values():
        base = ch_idxs[depth[n.id] % len(ch_idxs)]
        nxt = ch_idxs[(depth[n.id] + 1) % len(ch_idxs)]
        radios = [base, nxt] if n.radios >= 2 else [base]
        radio[n.id] = (radios + [ch_idxs[(depth[n.id] + 2) % len(ch_idxs)]])[: n.radios]
    link_ch = {}
    for lid, lk in net.links.items():
        shared = set(radio[lk.a]) & set(radio[lk.b])
        link_ch[lid] = min(shared) if shared else ch_idxs[0]
        if not shared:  # 强制共信道
            radio[lk.a][0] = link_ch[lid]
            radio[lk.b][0] = link_ch[lid]
    return _finish_plan(net, radio, link_ch, routes, loads, frame_slots)


def exhaustive_optimal(net: Network, ch_idxs=(1, 6, 11),
                       frame_slots: int = FRAME_SLOTS,
                       score_fn=None) -> tuple[Plan, float]:
    """小规模穷举:枚举每个节点的射频信道组合(路由同规划器首轮,
    调度器与规划器一致),取最差节点吞吐最优 —— 作为最优性上界参照."""
    from .planner import _plan_score
    routes = compute_routes(net)
    loads = link_loads(net, routes)
    nodes = sorted(net.nodes, key=lambda i: net.nodes[i].gateway, reverse=True)
    combos = []
    for r in range(1, max(n.radios for n in net.nodes.values()) + 1):
        combos.append(list(itertools.combinations(ch_idxs, min(r, len(ch_idxs)))))
    per_node_choices = []
    for nid in nodes:
        r = net.nodes[nid].radios
        opts = [list(c) for k in range(1, r + 1)
                for c in itertools.combinations(ch_idxs, min(k, len(ch_idxs)))]
        # 射频没用完等价于用完,只保留长度<=r 且去重;再补齐到 r 的重复不算新方案
        opts = [o + [o[-1]] * (r - len(o)) for o in opts]
        per_node_choices.append(sorted({tuple(o) for o in opts}))

    best, best_p, cnt = -1.0, None, 0
    for choice in itertools.product(*per_node_choices):
        radio = {nid: list(ch) for nid, ch in zip(nodes, choice)}
        link_ch = {}
        ok = True
        for lid, lk in net.links.items():
            shared = set(radio[lk.a]) & set(radio[lk.b])
            if not shared:
                ok = False
                break
            link_ch[lid] = min(shared)
        if not ok:
            continue
        cnt += 1
        p = _finish_plan(net, radio, link_ch, routes, loads, frame_slots)
        s = _plan_score(net, p, loads, frame_slots)
        if s > best:
            best, best_p = s, p
    return best_p, best

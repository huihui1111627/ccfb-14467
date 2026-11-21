"""验收仿真:按 SINR 逐时隙判定收发成败,输出端到端吞吐与时延.

与规划器内部估算不同,这里逐时隙、逐接收机用累加干扰重新判定
(规划器说什么不算,仿真里真能收到才算),并统计:
  - 每个杆子到出口的端到端吞吐(Mbps)与时延(ms)
  - 最差杆子
  - DFS 雷达事件:业务是否中断、切换后吞吐
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .model import Network, SINR_THRESHOLD_DB
from .planner import Plan, _link_between


@dataclass
class NodeResult:
    node: int
    throughput_mbps: float
    delay_ms: float
    hops: int


@dataclass
class SimResult:
    per_node: dict = field(default_factory=dict)   # node id -> NodeResult
    worst_node: int = -1
    worst_throughput: float = 0.0
    link_delivered: dict = field(default_factory=dict)  # lid -> Mbps 实际送达
    link_success_rate: dict = field(default_factory=dict)
    dfs_event: dict = field(default_factory=dict)  # 雷达事件统计


def _slot_success_map(net: Network, plan: Plan, slot_links,
                      link_channel) -> dict:
    """对一个周期内每个时隙做 SINR 判定,返回 link -> (成功时隙数, 总时隙数)."""
    chans = {c.idx: c for c in plan.channels}
    ok = defaultdict(int)
    tot = defaultdict(int)
    for s, lids in enumerate(slot_links):
        active = []
        for lid in lids:
            lk = net.links[lid]
            # 方向:朝出口(深度大的一端发)
            active.append((lid, lk.a, lk.b, link_channel[lid]))
        for lid, tx, rx, ci in active:
            interferers = [(t2, chans[c2]) for l2, t2, r2, c2 in active if l2 != lid]
            sinr = net.sinr_db(rx, tx, chans[ci], interferers)
            tot[lid] += 1
            if sinr >= SINR_THRESHOLD_DB:
                ok[lid] += 1
    return ok, tot


def simulate(net: Network, plan: Plan, frames: int = 20,
             radar: dict | None = None) -> SimResult:
    """跑 frames 个周期.

    radar: 可选 {触发帧号: [被雷达命中的信道 idx]} —— 命中后 DFS 射频
           立即切到 plan.backup_channels 的备用信道,统计业务连续性.
    """
    chans = {c.idx: c for c in plan.channels}
    frame_slots = len(plan.slot_links)
    slot_ms = 1.0

    # 当前生效的链路信道(雷达事件后会被改写);射频表用副本,不污染 plan
    cur_ch = dict(plan.link_channel)
    radios_now = {nid: list(r) for nid, r in plan.radio_channels.items()}
    # 链路方向:朝出口。由路由推出每个节点的下一跳
    next_hop = {}
    for src, path in plan.routes.items():
        for u, v in zip(path, path[1:]):
            next_hop[u] = v
    depth = _depths(net)

    link_ok_total = defaultdict(int)
    link_tot_total = defaultdict(int)
    delivered_bits = defaultdict(float)   # lid -> Mbps 累计
    dfs_log = {"events": [], "outage_slots": 0, "switched_radios": 0}

    for f in range(frames):
        if radar and f in radar:
            hit = radar[f]
            switched = 0
            # 链路级切换:整条链路切到预选备用信道,两端同步,业务不断
            for lid in list(cur_ch):
                if cur_ch[lid] in hit and chans[cur_ch[lid]].dfs:
                    bak = plan.link_backup.get(lid)
                    if bak is not None:
                        cur_ch[lid] = bak
                    else:
                        dfs_log["outage_slots"] += frame_slots  # 无备用,断链
            # 射频级切换(统计 + 保持射频表一致)
            for nid, radios in radios_now.items():
                for r, ci in enumerate(radios):
                    if ci in hit and chans[ci].dfs:
                        bak = plan.backup_channels.get(nid, {}).get(r)
                        if bak is not None:
                            radios[r] = bak
                            switched += 1
            dfs_log["events"].append({"frame": f, "hit_channels": hit,
                                      "switched_radios": switched})
            dfs_log["switched_radios"] += switched
        ok, tot = _slot_success_map(net, plan, plan.slot_links, cur_ch)
        for lid in tot:
            lk = net.links[lid]
            r = net.rate_mbps(net.sinr_db(lk.b, lk.a, chans[cur_ch[lid]], []))
            r = max(r, 6.0)
            link_ok_total[lid] += ok[lid]
            link_tot_total[lid] += tot[lid]
            delivered_bits[lid] += r * ok[lid] * slot_ms / 1000.0  # Mbit

    sim_time_s = frames * frame_slots * slot_ms / 1000.0
    link_mbps = {lid: delivered_bits[lid] / sim_time_s for lid in delivered_bits}
    link_sr = {lid: link_ok_total[lid] / link_tot_total[lid]
               for lid in link_tot_total if link_tot_total[lid] > 0}

    # 端到端:吞吐 = max-min 公平分配下该节点实际可得;时延 = 逐跳等时隙 + 传输
    from .planner import maxmin_fair
    flows = {}
    demands = {}
    for src, path in plan.routes.items():
        flows[src] = {_link_between(net, u, v) for u, v in zip(path, path[1:])}
        demands[src] = net.nodes[src].demand_mbps
    fair = maxmin_fair(flows, link_mbps, demands)
    res = SimResult(link_delivered=link_mbps, link_success_rate=link_sr,
                    dfs_event=dfs_log)
    for src, path in plan.routes.items():
        thr = fair.get(src, 0.0)
        delay = 0.0
        t = 0.0  # 包到达当前跳的时刻(周期内,ms)
        for u, v in zip(path, path[1:]):
            lid = _link_between(net, u, v)
            slots = sorted(plan.link_slots.get(lid, []))
            if not slots:
                delay += frames * frame_slots  # 根本排不上,惩罚
                continue
            pos = t % frame_slots
            nxt = next((s for s in slots if s >= pos), slots[0])
            wait = (nxt - pos) % frame_slots
            if wait == 0 and nxt < pos:
                wait = frame_slots
            delay += (wait + 1) * slot_ms  # 等到时隙 + 传一个时隙
            t += (wait + 1) * slot_ms
        hops = len(path) - 1
        res.per_node[src] = NodeResult(src, thr, delay, hops)
    if res.per_node:
        worst = min(res.per_node.values(), key=lambda r: r.throughput_mbps)
        res.worst_node, res.worst_throughput = worst.node, worst.throughput_mbps
    return res


def _depths(net):
    from collections import deque
    dist = {n.id: (0 if n.gateway else math.inf) for n in net.nodes.values()}
    dq = deque([n.id for n in net.nodes.values() if n.gateway])
    while dq:
        u = dq.popleft()
        for lid in net.adj[u]:
            v = net.links[lid].other(u)
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist

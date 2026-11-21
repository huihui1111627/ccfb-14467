"""联合优化规划器:路由 × 信道分配 × 时隙调度 迭代求解 + DFS 雷达备用方案.

流程(不是先定路由再配信道,而是联合迭代):
  1. 以当前链路代价(容量感知)为每个节点选到出口的路径 -> 链路负载
  2. 按负载做射频级信道分配(每杆射频数有限,链路两端须共信道)
  3. 在物理干扰模型(干扰累加)下做加权时隙调度,时隙顺序顺流量方向
  4. 用调度得到的链路容量更新路由代价,回到 1,迭代数轮取最优
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .model import (Channel, Network, SINR_THRESHOLD_DB, default_channels,
                    overlap_factor)

FRAME_SLOTS = 32          # 一个调度周期的时隙数
SLOT_MS = 1.0             # 时隙长度
JOINT_ITERS = 4


# ---------------------------------------------------------------- 结果结构
@dataclass
class Plan:
    net: Network
    radio_channels: dict          # node -> [channel idx per radio]
    link_channel: dict            # link_id -> channel idx
    routes: dict                  # node -> [node,...,gateway]
    slot_links: list              # slot -> [link_id]  (方向: a->b 朝出口)
    link_slots: dict = field(default_factory=dict)   # link_id -> [slot]
    backup_channels: dict = field(default_factory=dict)  # node -> {radio: backup ch idx}
    link_backup: dict = field(default_factory=dict)      # lid -> 雷达时切到的备用信道
    channels: list = field(default_factory=default_channels)

    def chan_of(self, ch_idx):
        return next(c for c in self.channels if c.idx == ch_idx)


def maxmin_fair(flows: dict, caps: dict, demands: dict) -> dict:
    """max-min 公平分配:flows={节点: 路径链路集}, caps={链路: 容量},
    demands={节点: 需求}. 返回每节点实际可得吞吐."""
    rem = dict(caps)
    active = {n: set(ls) for n, ls in flows.items()}
    res = {}
    while active:
        users = defaultdict(int)
        for ls in active.values():
            for l in ls:
                users[l] += 1
        best_share, best_link = math.inf, None
        for l, u in users.items():
            share = rem.get(l, 0.0) / u
            if share < best_share:
                best_share, best_link = share, l
        for nid in [n for n, ls in active.items() if best_link in ls]:
            r = min(best_share, demands.get(nid, math.inf))
            res[nid] = r
            for l in active[nid]:
                rem[l] = rem.get(l, 0.0) - r
            del active[nid]
    return res


# ---------------------------------------------------------------- 路由
def _k_shortest_paths(net: Network, src: int, gw: int, k: int = 3,
                      cost=None) -> list[list[int]]:
    """简单 k 短路(按代价),用于路由调整的候选路径集."""
    if cost is None:
        cost = {lid: 1.0 for lid in net.links}
    paths = []
    # 延边 Dijkstra 的 k 短路(网络小,直接 DFS 限深)
    stack = [(src, [src], 0.0)]
    hops_limit = len(net.nodes)
    while stack and len(paths) < k * 6:
        u, path, c = stack.pop()
        if u == gw:
            paths.append((c, path))
            continue
        if len(path) > hops_limit:
            continue
        for lid in net.adj[u]:
            lk = net.links[lid]
            v = lk.other(u)
            if v in path:
                continue
            stack.append((v, path + [v], c + cost[lid]))
    paths.sort(key=lambda t: t[0])
    out, seen = [], set()
    for c, p in paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
        if len(out) >= k:
            break
    return out


def compute_routes(net: Network, cost=None) -> dict:
    gws = [n.id for n in net.nodes.values() if n.gateway]
    routes = {}
    for n in net.nodes.values():
        if n.gateway:
            continue
        best, best_c = None, math.inf
        for gw in gws:
            ps = _k_shortest_paths(net, n.id, gw, k=1, cost=cost)
            if ps and (c := _path_cost(net, ps[0], cost)) < best_c:
                best, best_c = ps[0], c
        if best is None:
            raise RuntimeError(f"节点 {n.id} 到不了任何出口")
        routes[n.id] = best
    return routes


def _path_cost(net, path, cost):
    if cost is None:
        return len(path) - 1
    s = 0.0
    for u, v in zip(path, path[1:]):
        lid = _link_between(net, u, v)
        s += cost[lid]
    return s


def _link_between(net, u, v):
    for lid in net.adj[u]:
        if net.links[lid].other(u) == v:
            return lid
    return None


def link_loads(net: Network, routes: dict) -> dict:
    """每条链路承担的上行流量(Mbps)."""
    load = defaultdict(float)
    for src, path in routes.items():
        d = net.nodes[src].demand_mbps
        for u, v in zip(path, path[1:]):
            load[_link_between(net, u, v)] += d
    return dict(load)


# ---------------------------------------------------------------- 信道分配
def assign_channels(net: Network, loads: dict,
                    channels: list[Channel]) -> tuple[dict, dict]:
    """射频级贪心信道分配.

    约束:每杆射频数固定;链路两端必须各有一个射频在同一信道.
    目标:高负载链路尽量落在互不重叠、空间上互相隔离的信道 -> 冲突图稀疏.
    允许使用部分重叠信道(干扰按 overlap_factor 折算),DFS 信道也可用(另有备用).
    """
    chans = {c.idx: c for c in channels}
    radio = {n.id: [] for n in net.nodes.values()}      # node -> [ch idx]
    link_ch = {}
    # 出口射频先铺满最不重叠的信道,作为全网信道骨架
    non_dfs = [c for c in channels if not c.dfs]
    orthogonal = _max_independent_set(channels)
    for n in net.nodes.values():
        if n.gateway:
            for c in orthogonal[: n.radios]:
                radio[n.id].append(c.idx)

    def node_has(u, ci):
        return ci in radio[u]

    def add_radio(u, ci):
        if len(radio[u]) < net.nodes[u].radios:
            radio[u].append(ci)
            return True
        return False

    def interference_cost(lid, ci):
        """把链路 lid 放到信道 ci 上引入的加权冲突(负载 × 重叠 × 空间邻近)."""
        lk = net.links[lid]
        cost = 0.0
        for lid2, ci2 in link_ch.items():
            f = overlap_factor(chans[ci], chans[ci2])
            if f <= 0:
                continue
            lk2 = net.links[lid2]
            # 两端对接收端的干扰取最大(近似冲突强度)
            w = 0.0
            for tx, rx in ((lk2.a, lk.b), (lk2.b, lk.a)):
                sig = net.rx_power_dbm(lk.a, lk.b)
                itf = net.rx_power_dbm(tx, rx)
                w = max(w, f * 10 ** ((itf - sig) / 10))
            cost += (loads.get(lid, 0) + loads.get(lid2, 0)) * w
        return cost

    # 按负载从大到小给链路上信道
    for lid in sorted(net.links, key=lambda l: -loads.get(l, 0)):
        lk = net.links[lid]
        shared = [ci for ci in set(radio[lk.a]) & set(radio[lk.b])]
        if shared:
            link_ch[lid] = min(shared, key=lambda ci: interference_cost(lid, ci))
            continue
        best = None
        # 候选:两端已有信道的任意组合 + 给空射频开新信道
        cand = set(radio[lk.a]) | set(radio[lk.b])
        empt_a = net.nodes[lk.a].radios - len(radio[lk.a])
        empt_b = net.nodes[lk.b].radios - len(radio[lk.b])
        if empt_a > 0 or empt_b > 0:
            cand |= {c.idx for c in channels}
        for ci in cand:
            ok_a, ok_b = node_has(lk.a, ci), node_has(lk.b, ci)
            if not ok_a and not (empt_a > 0):
                continue
            if not ok_b and not (empt_b > 0):
                continue
            c = interference_cost(lid, ci)
            # DFS 信道加一点风险折价(可能被雷达赶走)
            if chans[ci].dfs:
                c *= 1.15
            if best is None or c < best[0]:
                best = (c, ci)
        if best is None:
            # 射频占满且无共信道:把负载最轻端点的一个射频改成对方已有信道
            ci = max(set(radio[lk.a]) | set(radio[lk.b]),
                     key=lambda c: -interference_cost(lid, c))
            for u in (lk.a, lk.b):
                if ci not in radio[u]:
                    victim = min(range(len(radio[u])),
                                 key=lambda r: _radio_load(net, radio, link_ch, u, r, loads))
                    radio[u][victim] = ci
            link_ch[lid] = ci
        else:
            _, ci = best
            if not node_has(lk.a, ci):
                add_radio(lk.a, ci)
            if not node_has(lk.b, ci):
                add_radio(lk.b, ci)
            link_ch[lid] = ci

    # 修复:射频被抢占改信道后,两端不再共信道的链路重新协商。
    # 最小破坏贪心:每次修负载最高的断链,候选动作 = 在某一端放一个
    # 对方已有的信道,破坏 = 被挤掉的射频上还在用旧信道的链路数。
    for _ in range(8 * len(net.links) + 10):
        broken = [lid for lid, ci in link_ch.items()
                  if ci not in radio[net.links[lid].a] or ci not in radio[net.links[lid].b]]
        if not broken:
            break
        lid = max(broken, key=lambda l: loads.get(l, 0.0))
        lk = net.links[lid]
        shared = [c for c in set(radio[lk.a]) & set(radio[lk.b])]
        if shared:
            link_ch[lid] = min(shared, key=lambda c: interference_cost(lid, c))
            continue
        best = None  # (破坏数, 动作)
        for u, w in ((lk.a, lk.b), (lk.b, lk.a)):
            for ci2 in set(radio[w]) | {link_ch[lid]}:
                if ci2 in radio[u]:
                    continue
                for r in range(len(radio[u])):
                    old = radio[u][r]
                    damage = sum(1 for l2 in net.adj[u]
                                 if l2 != lid and link_ch.get(l2) == old)
                    key = (damage, _radio_load(net, radio, link_ch, u, r, loads))
                    if best is None or key < best[0]:
                        best = (key, (u, r, ci2))
        if best is not None:
            u, r, ci2 = best[1]
            radio[u][r] = ci2
            if ci2 in radio[net.links[lid].a] and ci2 in radio[net.links[lid].b]:
                link_ch[lid] = ci2
    # 兜底(理论上到不了):全网补一个公共信道,剩余断链全部退到公共信道
    broken = [lid for lid, ci in link_ch.items()
              if ci not in radio[net.links[lid].a] or ci not in radio[net.links[lid].b]]
    if broken:
        common = 1  # 2.4G 信道 1 作为全网公共保底信道
        for n in net.nodes.values():
            if common not in radio[n.id]:
                radio[n.id][0] = common
        for lid in list(link_ch):
            lk = net.links[lid]
            ci = link_ch[lid]
            if ci not in radio[lk.a] or ci not in radio[lk.b]:
                shared = set(radio[lk.a]) & set(radio[lk.b])
                link_ch[lid] = min(shared) if shared else common

    # 没分满的射频补一个正交信道,避免浪费
    for n in net.nodes.values():
        for c in orthogonal:
            if len(radio[n.id]) >= n.radios:
                break
            if c.idx not in radio[n.id]:
                radio[n.id].append(c.idx)
    return radio, link_ch


def _radio_load(net, radio, link_ch, u, r, loads):
    """节点 u 第 r 个射频当前承担的链路负载(用于抢占时挑最轻的牺牲者)."""
    ci = radio[u][r]
    s = 0.0
    for lid in net.adj[u]:
        if link_ch.get(lid) == ci:
            s += loads.get(lid, 0.0)
    return s


def _max_independent_set(channels: list[Channel]) -> list[Channel]:
    """从信道集里挑一组互不重叠的(贪心,优先非 DFS)."""
    out = []
    for c in sorted(channels, key=lambda c: (c.dfs, c.idx)):
        if all(overlap_factor(c, o) < 0.05 for o in out):
            out.append(c)
    return out


# ---------------------------------------------------------------- 冲突图
def build_conflict_graph(net: Network, link_ch: dict,
                         channels: list[Channel]) -> dict:
    """两两冲突边(边权=单方干扰是否已压垮 SINR 门限).

    注意:真正的调度判定不只看两两——schedule_slots 里用的是
    累加干扰的 SINR 检验,这张图用于分析和调度时的候选剪枝.
    """
    chans = {c.idx: c for c in channels}
    edges = defaultdict(dict)
    lids = list(link_ch)
    for i, l1 in enumerate(lids):
        for l2 in lids[i + 1:]:
            w = _pairwise_conflict(net, l1, l2, chans[link_ch[l1]],
                                   chans[link_ch[l2]])
            if w > 0:
                edges[l1][l2] = w
                edges[l2][l1] = w
    return dict(edges)


def _pairwise_conflict(net, l1, l2, c1, c2) -> float:
    f = overlap_factor(c1, c2)
    if f <= 0:
        return 0.0
    lk1, lk2 = net.links[l1], net.links[l2]
    worst = 0.0
    # 四个 发射->接收 干扰方向
    for tx_lk, rx_lk, ctx, crx in ((lk1, lk2, c1, c2), (lk2, lk1, c2, c1)):
        for tx, rx in ((tx_lk.a, rx_lk.a), (tx_lk.a, rx_lk.b),
                       (tx_lk.b, rx_lk.a), (tx_lk.b, rx_lk.b)):
            sig_tx = rx_lk.a if rx == rx_lk.b else rx_lk.b
            sinr = net.sinr_db(rx, sig_tx, crx, [(tx, ctx)])
            if sinr < SINR_THRESHOLD_DB:
                worst = max(worst, SINR_THRESHOLD_DB - sinr)
    return worst


# ---------------------------------------------------------------- 时隙调度
def schedule_slots(net: Network, link_ch: dict, loads: dict,
                   channels: list[Channel], radio: dict,
                   frame_slots: int = FRAME_SLOTS) -> tuple[list, dict]:
    """加权 TDMA 调度.

    - 每条链路所需时隙数 ∝ 负载 / 干净信道速率(流量匹配)
    - 每个时隙选一组互不冲突的链路:判定用累加干扰 SINR(不是只看两两)
    - 射频约束:同一时隙一个射频只能干一件事;同节点同信道不能又收又发
    - 时隙顺序顺流量方向:离出口远的链路先排(包在一个周期内逐级汇入)
    """
    chans = {c.idx: c for c in channels}
    # 每条链路所需时隙数:容量 = 速率 × (时隙数/周期) ≥ 负载  =>  时隙数 ≥ 负载/速率 × 周期
    need = {}
    for lid, ld in loads.items():
        r = max(6.0, net.clean_rate_mbps(lid, chans[link_ch[lid]]))
        need[lid] = max(1, math.ceil(ld / r * frame_slots))

    # 深度 = 到出口的跳数(经路由),用于顺流排序
    depth = _link_depth(net, loads)

    slot_links = [[] for _ in range(frame_slots)]
    slot_radio_used = [defaultdict(set) for _ in range(frame_slots)]  # slot -> node -> {radio idx}
    link_slots = defaultdict(list)

    order = sorted(need, key=lambda l: (-depth[l], -loads[l]))

    def try_place(lid, s):
        lk = net.links[lid]
        ci = link_ch[lid]
        ra = radio[lk.a].index(ci)
        rb = radio[lk.b].index(ci)
        if _slot_feasible(net, slot_links[s], slot_radio_used[s],
                          lid, ci, ra, rb, link_ch, chans):
            slot_links[s].append(lid)
            slot_radio_used[s][lk.a].add(ra)
            slot_radio_used[s][lk.b].add(rb)
            link_slots[lid].append(s)
            return True
        return False

    # 第一遍:每条承载链路保底 1 个时隙(链路数 ≤ 周期时隙数时必然成功)
    for i, lid in enumerate(order):
        try_place(lid, i % frame_slots)
    # 第二遍:加权轮转填充 —— 每轮每条链路再分 1 个时隙,直到满足需求。
    # 轮转而不是一次拿完,时隙在周期内分布更均匀,也避免深度大的链路抢光好位置。
    remaining = {lid: need[lid] - len(link_slots.get(lid, [])) for lid in order}
    for _round in range(frame_slots):
        if all(v <= 0 for v in remaining.values()):
            break
        progress = False
        for lid in order:
            if remaining[lid] <= 0:
                continue
            # 顺流:优先选比上一跳更早的时隙 —— 从 0 开始找最早可行槽
            for s in range(frame_slots):
                if s in link_slots[lid]:
                    continue
                if try_place(lid, s):
                    remaining[lid] -= 1
                    progress = True
                    break
        if not progress:
            break
    # 修复:一个时隙都没抢到的链路,从驱逐代价最小的时隙里挤出位置
    for lid in order:
        if link_slots.get(lid):
            continue
        lk = net.links[lid]
        ci = link_ch[lid]
        ra = radio[lk.a].index(ci)
        rb = radio[lk.b].index(ci)
        best = None  # (驱逐代价, slot, evict list)
        for s in sorted(range(frame_slots), key=lambda s: len(slot_links[s])):
            evict = [l2 for l2 in slot_links[s]
                     if not _coexist_ok(net, l2, lid, link_ch, chans, radio)]
            trial = [l2 for l2 in slot_links[s] if l2 not in evict]
            trial_map = _radio_map(net, trial, link_ch, radio)
            if _slot_feasible(net, trial, trial_map, lid, ci, ra, rb,
                              link_ch, chans):
                cost = sum(need.get(l2, 1) for l2 in evict)
                if best is None or cost < best[0]:
                    best = (cost, s, evict)
        if best is None:
            continue
        _, s, evict = best
        for l2 in evict:
            slot_links[s].remove(l2)
            link_slots[l2].remove(s)
        slot_radio_used[s] = _radio_map(net, slot_links[s], link_ch, radio)
        slot_links[s].append(lid)
        slot_radio_used[s][lk.a].add(ra)
        slot_radio_used[s][lk.b].add(rb)
        link_slots[lid] = [s]
        # 被逐链路重新找位置
        for l2 in evict:
            lk2 = net.links[l2]
            ci2 = link_ch[l2]
            ra2 = radio[lk2.a].index(ci2)
            rb2 = radio[lk2.b].index(ci2)
            for s2 in range(frame_slots):
                if _slot_feasible(net, slot_links[s2], slot_radio_used[s2],
                                  l2, ci2, ra2, rb2, link_ch, chans):
                    slot_links[s2].append(l2)
                    slot_radio_used[s2][lk2.a].add(ra2)
                    slot_radio_used[s2][lk2.b].add(rb2)
                    link_slots[l2].append(s2)
                    break
    return slot_links, dict(link_slots)


def _coexist_ok(net, l2, lid, link_ch, chans, radio) -> bool:
    """两条链路能否同槽共存(粗判:射频占用 + 半双工 + 两两 SINR);
    最终仍以 _slot_feasible 的累加干扰判定为准."""
    lk, lk2 = net.links[lid], net.links[l2]
    ci, ci2 = link_ch[lid], link_ch[l2]
    for u in (lk.a, lk.b):
        if u in (lk2.a, lk2.b):
            if radio[u].index(ci) == radio[u].index(ci2):
                return False            # 同一射频
            if ci == ci2:
                return False            # 同节点同信道又收又发
    return _pairwise_conflict(net, l2, lid, chans[ci2], chans[ci]) <= 0


def _radio_map(net, links, link_ch, radio):
    m = defaultdict(set)
    for l2 in links:
        lk = net.links[l2]
        ci = link_ch[l2]
        m[lk.a].add(radio[lk.a].index(ci))
        m[lk.b].add(radio[lk.b].index(ci))
    return m


def _link_depth(net, loads):
    """链路深度:发端(远离出口端)到出口的跳数.由路由负载方向推断."""
    depth = {}
    # 用 BFS 从出口算节点跳数
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
    for lid in loads:
        lk = net.links[lid]
        depth[lid] = max(dist[lk.a], dist[lk.b])
    return depth


def _slot_feasible(net, cur_links, radio_used, lid, ci, ra, rb,
                   link_ch, chans) -> bool:
    """把 lid 加入当前时隙是否可行:射频占用 + 半双工 + 累加干扰 SINR."""
    lk = net.links[lid]
    # 射频占用:一个射频一个时隙只能参与一条链路
    if ra in radio_used.get(lk.a, set()) or rb in radio_used.get(lk.b, set()):
        return False
    # 半双工/自干扰:同节点同一时隙不能既发又收(不同射频同信道自干扰也禁止)
    busy_nodes = set()
    node_chan = defaultdict(set)
    for l2 in cur_links:
        lk2 = net.links[l2]
        busy_nodes.add(lk2.a)
        busy_nodes.add(lk2.b)
        node_chan[lk2.a].add(link_ch[l2])
        node_chan[lk2.b].add(link_ch[l2])
    if lk.a in busy_nodes and ci in node_chan[lk.a]:
        return False
    if lk.b in busy_nodes and ci in node_chan[lk.b]:
        return False
    # 累加干扰 SINR:新链路加入后,时隙内所有接收都须过门限
    active = [(net.links[l2].a, net.links[l2].b, link_ch[l2]) for l2 in cur_links]
    active.append((lk.a, lk.b, ci))
    for tx, rx, c in active:
        interferers = [(t2, chans[c2]) for t2, r2, c2 in active if t2 != tx]
        if net.sinr_db(rx, tx, chans[c], interferers) < SINR_THRESHOLD_DB:
            return False
    return True


# ---------------------------------------------------------------- DFS 雷达备用
def plan_dfs_backup(net: Network, plan: Plan) -> tuple[dict, dict]:
    """DFS 雷达备用方案.

    链路级:每条用 DFS 信道的链路预选一个非 DFS 备用信道,雷达来时
    整条链路立即切过去 —— 两端同步切换,业务不断。优先选两端射频
    已有的信道(零改动),其次一端已有的。
    射频级(展示用):该射频上多数链路选的备用信道。
    返回 (radio_backup, link_backup)。
    """
    chans = {c.idx: c for c in plan.channels}
    non_dfs = [c for c in plan.channels if not c.dfs]
    link_backup = {}
    for lid, ci in plan.link_channel.items():
        if not chans[ci].dfs:
            continue
        lk = net.links[lid]
        my_slots = plan.link_slots.get(lid, [])
        best, best_key = None, None
        for cand in non_dfs:
            in_a = cand.idx in plan.radio_channels[lk.a]
            in_b = cand.idx in plan.radio_channels[lk.b]
            # 关键:在该链路已分到的每个时隙里预演 —— 切成 cand 后,
            # 同槽所有链路的累加 SINR 是否仍过门限(切换后业务不断)
            bad_slots = 0
            for s in my_slots:
                active = []
                for l2 in plan.slot_links[s]:
                    lk2 = net.links[l2]
                    c2 = cand.idx if l2 == lid else plan.link_channel[l2]
                    active.append((lk2.a, lk2.b, c2))
                for tx, rx, c in active:
                    interferers = [(t2, chans[c2]) for t2, r2, c2 in active
                                   if not (t2 == tx and r2 == rx)]
                    if net.sinr_db(rx, tx, chans[c], interferers) < SINR_THRESHOLD_DB:
                        bad_slots += 1
                        break
            key = (-bad_slots, in_a + in_b)
            if best_key is None or key > best_key:
                best, best_key = cand.idx, key
        link_backup[lid] = best
    radio_backup = {}
    for nid, radios in plan.radio_channels.items():
        for r, ci in enumerate(radios):
            if not chans[ci].dfs:
                continue
            votes = defaultdict(int)
            for lid in net.adj[nid]:
                if lid in link_backup:
                    votes[link_backup[lid]] += 1
            if votes:
                radio_backup.setdefault(nid, {})[r] = max(votes, key=votes.get)
    return radio_backup, link_backup


# ---------------------------------------------------------------- 联合优化主循环
def plan_network(net: Network, channels: list[Channel] | None = None,
                 iters: int = JOINT_ITERS, frame_slots: int = FRAME_SLOTS) -> Plan:
    channels = channels or default_channels()
    best, best_score = None, -1.0
    cost = None
    for it in range(iters):
        routes = compute_routes(net, cost)
        loads = link_loads(net, routes)
        radio, link_ch = assign_channels(net, loads, channels)
        slot_links, link_slots = schedule_slots(net, link_ch, loads,
                                                channels, radio, frame_slots)
        plan = Plan(net=net, radio_channels=radio, link_channel=link_ch,
                    routes=routes, slot_links=slot_links,
                    link_slots=link_slots, channels=channels)
        plan.backup_channels, plan.link_backup = plan_dfs_backup(net, plan)
        # 评分:全网保证吞吐(按各节点瓶颈链路容量的最小值加总)
        score = _plan_score(net, plan, loads, frame_slots)
        if score > best_score:
            best, best_score = plan, score
        # 用本轮容量更新路由代价 -> 下一轮路由避开拥塞链路(联合优化闭环)
        cost = {}
        for lid in net.links:
            cap = _link_capacity(net, plan, lid, frame_slots)
            ld = loads.get(lid, 0.0)
            cost[lid] = 1.0 + 8.0 * (ld / cap if cap > 0 else 10.0)
    return best


def _link_capacity(net, plan, lid, frame_slots):
    if lid not in plan.link_channel:
        return 0.0
    ch = plan.chan_of(plan.link_channel[lid])
    r = max(6.0, net.clean_rate_mbps(lid, ch))
    return r * len(plan.link_slots.get(lid, [])) / frame_slots


def _plan_score(net, plan, loads, frame_slots):
    """最差节点的 max-min 公平吞吐 —— 优化目标就是最差的杆子."""
    caps = {lid: _link_capacity(net, plan, lid, frame_slots) for lid in net.links}
    flows = {}
    demands = {}
    for src, path in plan.routes.items():
        flows[src] = {_link_between(net, u, v) for u, v in zip(path, path[1:])}
        demands[src] = net.nodes[src].demand_mbps
    fair = maxmin_fair(flows, caps, demands)
    return min(fair.values()) if fair else 0.0

"""端到端演示:山区景区场景 -> 规划 -> 验收仿真 -> 基线对比 -> 小规模穷举 -> HTML 报告.

用法: python3 run_demo.py
输出: report.html(主场景), 并在终端打印对比表
"""
from __future__ import annotations

import math
import random

from meshnet.model import Network, Node, Terrain, default_channels
from meshnet.planner import plan_network
from meshnet.simulate import simulate
from meshnet.baselines import (baseline_alternate, baseline_single_channel,
                               exhaustive_optimal)
from meshnet.report import render_html


def build_scenic_area(seed: int = 19) -> Network:
    """山区景区:沿谷地游步道布 14 根杆,两端山脚各一个有线出口."""
    for attempt in range(50):
        rng = random.Random(seed + attempt)
        terrain = Terrain.synthetic_hills(n=80, size_m=4000.0)
        nodes = [
            Node(0, 300, 300, radios=3, gateway=True),
            Node(1, 3700, 3700, radios=3, gateway=True),
        ]
        # 沿对角线谷地撒杆,带扰动;流量大的在景点密集处
        for i in range(2, 16):
            t = (i - 1) / 15.0
            x = 300 + t * 3400 + rng.uniform(-260, 260)
            y = 300 + t * 3400 + rng.uniform(-260, 260)
            demand = rng.choice([16.0, 24.0, 32.0, 40.0])  # 监控路数不同,高峰压力大
            nodes.append(Node(i, x, y, radios=rng.choice([2, 2, 3]),
                              demand_mbps=demand))
        net = Network(terrain, nodes, default_channels())
        if _fully_connected(net):
            return net
    raise RuntimeError("生成不出连通的场景")


def _fully_connected(net: Network) -> bool:
    """每个杆都至少有一条路径到某个出口."""
    seen = {n.id for n in net.nodes.values() if n.gateway}
    frontier = list(seen)
    while frontier:
        u = frontier.pop()
        for lid in net.adj[u]:
            v = net.links[lid].other(u)
            if v not in seen:
                seen.add(v)
                frontier.append(v)
    return len(seen) == len(net.nodes)


def build_small(seed: int = 3) -> Network:
    """小规模(6 杆 1 出口),用于和穷举最优对比."""
    for attempt in range(50):
        rng = random.Random(seed + attempt)
        terrain = Terrain.synthetic_hills(n=60, size_m=2500.0, seed=5)
        nodes = [Node(0, 400, 400, radios=2, gateway=True)]
        for i in range(1, 6):
            t = i / 6.0
            nodes.append(Node(i, 400 + t * 1600 + rng.uniform(-200, 200),
                              400 + t * 1600 + rng.uniform(-200, 200),
                              radios=2, demand_mbps=8.0))
        net = Network(terrain, nodes, default_channels())
        if _fully_connected(net):
            return net
    raise RuntimeError("生成不出连通的小场景")


def show(name, sim):
    avg_t = sum(r.throughput_mbps for r in sim.per_node.values()) / max(1, len(sim.per_node))
    avg_d = sum(r.delay_ms for r in sim.per_node.values()) / max(1, len(sim.per_node))
    print(f"  {name:<14} 最差杆 {sim.worst_node:>2}  最差吞吐 {sim.worst_throughput:6.2f} Mbps"
          f"  平均吞吐 {avg_t:6.2f}  平均时延 {avg_d:6.1f} ms")


def main():
    print("=" * 72)
    print("主场景:山区景区 16 杆 / 2 出口,联合优化 vs 两种经验做法")
    print("=" * 72)
    net = build_scenic_area()
    print(f"链路数 {len(net.links)},信道 {len(net.channels)} 个"
          f"(其中 DFS {sum(c.dfs for c in net.channels)} 个)")

    plan = plan_network(net)

    # 稳态(无雷达)—— 方案对比用这个
    sim_main = simulate(net, plan)
    p_single = baseline_single_channel(net)
    s_single = simulate(net, p_single)
    p_alt = baseline_alternate(net)
    s_alt = simulate(net, p_alt)

    print("稳态(无雷达事件):")
    show("联合优化(本方案)", sim_main)
    show("全网同一信道", s_single)
    show("隔跳换信道", s_alt)

    # DFS 雷达事件:第 5 帧命中两个 DFS 信道
    radar = {5: [52, 56]}
    sim_radar = simulate(net, plan, radar=radar)
    sim_post = simulate(net, plan, radar={0: [52, 56]})  # 全程备用信道 = 切换后稳态
    ev = sim_radar.dfs_event["events"][0]
    print(f"\nDFS 雷达事件:第 {ev['frame']} 帧命中 {ev['hit_channels']},"
          f"切换射频 {ev['switched_radios']} 个,断链时隙 {sim_radar.dfs_event['outage_slots']}")
    print(f"  切换前最差吞吐 {sim_main.worst_throughput:.2f} Mbps -> "
          f"切换后稳态 {sim_post.worst_throughput:.2f} Mbps(业务不中断,降级运行)")

    print()
    print("=" * 72)
    print("小规模场景:6 杆 / 1 出口,与穷举最优对比")
    print("=" * 72)
    snet = build_small()
    sp = plan_network(snet)
    ss = simulate(snet, sp)
    show("联合优化(本方案)", ss)
    opt_p, opt_score = exhaustive_optimal(snet)
    s_opt = simulate(snet, opt_p)
    show("穷举最优", s_opt)
    gap = (1 - ss.worst_throughput / s_opt.worst_throughput) * 100 if s_opt.worst_throughput else 0
    exhaustive_note = (f"小规模(6 杆)穷举最优最差吞吐 {s_opt.worst_throughput:.2f} Mbps,"
                       f"本方案 {ss.worst_throughput:.2f} Mbps,差距 {gap:.1f}%。")
    print(f"  -> {exhaustive_note}")

    html = render_html(
        net,
        {"联合优化": plan, "全网同一信道": p_single, "隔跳换信道": p_alt},
        {"联合优化": sim_main, "全网同一信道": s_single, "隔跳换信道": s_alt},
        "联合优化", sim_radar.dfs_event, exhaustive_note,
        dfs_pre=sim_main.worst_throughput, dfs_post=sim_post.worst_throughput)
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("\n报告已写入 report.html(浏览器打开可看地形图/链路信道/时隙表)")


if __name__ == "__main__":
    main()

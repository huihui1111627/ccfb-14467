"""地形、网络模型、信道重叠模型与传播(距离 + 地形遮挡)计算."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- 物理常量
P_TX_DBM = 23.0          # 发射功率
NOISE_DBM = -95.0        # 底噪
ANTENNA_GAIN_DB = 8.0    # 定向天线增益(收发各计一次)
SINR_THRESHOLD_DB = 8.0  # 最低可用 MCS 对应的 SINR 门限
BW_MHZ = 20.0
RATE_CAP_MBPS = 150.0
POLE_HEIGHT_M = 6.0      # 杆高(天线离地)


# ---------------------------------------------------------------- 地形
class Terrain:
    """高程栅格,双线性插值;沿线采样判断通视并折算遮挡损耗."""

    def __init__(self, elev: np.ndarray, size_m: float):
        self.elev = np.asarray(elev, dtype=float)
        self.size_m = float(size_m)
        self.n = self.elev.shape[0]

    @classmethod
    def synthetic_hills(cls, n: int = 80, size_m: float = 4000.0, seed: int = 7):
        """合成山地:几座高斯山头叠加,中间一条谷地方便布杆."""
        rng = np.random.default_rng(seed)
        xs = np.linspace(0, 1, n)
        gx, gy = np.meshgrid(xs, xs)
        elev = np.zeros((n, n))
        hills = [(0.25, 0.30, 0.16, 260), (0.72, 0.25, 0.13, 300),
                 (0.55, 0.62, 0.18, 340), (0.20, 0.80, 0.14, 240),
                 (0.85, 0.75, 0.12, 280)]
        for cx, cy, s, h in hills:
            elev += h * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s * s))
        elev += rng.normal(0, 4.0, (n, n))
        # 谷地:沿对角线压出一条走廊
        diag = np.abs(gx - gy)
        elev *= (0.35 + 0.65 * np.clip(diag / 0.25, 0, 1))
        return cls(elev, size_m)

    def height(self, x_m: float, y_m: float) -> float:
        n, s = self.n, self.size_m
        fx = np.clip(x_m / s * (n - 1), 0, n - 1.001)
        fy = np.clip(y_m / s * (n - 1), 0, n - 1.001)
        x0, y0 = int(fx), int(fy)
        dx, dy = fx - x0, fy - y0
        e = self.elev
        return float(e[y0, x0] * (1 - dx) * (1 - dy) + e[y0, x0 + 1] * dx * (1 - dy)
                     + e[y0 + 1, x0] * (1 - dx) * dy + e[y0 + 1, x0 + 1] * dx * dy)

    def obstruction_loss_db(self, a: "Node", b: "Node", samples: int = 40) -> float:
        """视线被地形遮挡的附加损耗:按遮挡深度折算(类刃形绕射的简化)."""
        d = math.dist((a.x, a.y), (b.x, b.y))
        if d < 1:
            return 0.0
        ha = self.height(a.x, a.y) + POLE_HEIGHT_M
        hb = self.height(b.x, b.y) + POLE_HEIGHT_M
        worst = 0.0
        for i in range(1, samples):
            t = i / samples
            x, y = a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t
            los_h = ha + (hb - ha) * t
            # 第一菲涅尔区半径,地形侵入菲涅尔区即产生损耗
            r_fresnel = 8.66 * math.sqrt((d * t * (1 - t)) / d / 2.4) if d > 0 else 0
            block = self.height(x, y) - (los_h - 0.6 * r_fresnel)
            worst = max(worst, block)
        if worst <= 0:
            return 0.0
        return min(35.0, 6.0 + 9.0 * math.log10(1.0 + worst))  # 遮挡越深损耗越大


# ---------------------------------------------------------------- 信道
@dataclass(frozen=True)
class Channel:
    idx: int
    freq_mhz: float
    dfs: bool = False      # 雷达频段,需动态频谱避让


def default_channels() -> list[Channel]:
    """2.4G 1..11(5MHz 间隔,只有 1/6/11 完全不重叠) + 4 个 5G 信道(其中两个 DFS)."""
    chs = [Channel(i, 2412 + 5 * (i - 1)) for i in range(1, 12)]
    chs += [Channel(36, 5180), Channel(40, 5200),
            Channel(52, 5260, dfs=True), Channel(56, 5280, dfs=True)]
    return chs


def overlap_factor(ca: Channel, cb: Channel) -> float:
    """部分重叠信道干扰折算:2.4G 按 22MHz 占用带宽线性衰减;跨频段为 0."""
    ba = ca.freq_mhz < 3000
    bb = cb.freq_mhz < 3000
    if ba != bb:
        return 0.0
    sep = abs(ca.freq_mhz - cb.freq_mhz)
    if ba:
        return max(0.0, 1.0 - sep / 22.0)
    return 1.0 if sep < 20 else 0.0


# ---------------------------------------------------------------- 网络
@dataclass
class Node:
    id: int
    x: float
    y: float
    radios: int = 2
    gateway: bool = False
    demand_mbps: float = 8.0   # 该杆业务(视频监控等)上行流量


@dataclass
class Link:
    id: int
    a: int
    b: int
    dist_m: float
    path_loss_db: float

    def other(self, n: int) -> int:
        return self.b if n == self.a else self.a


class Network:
    def __init__(self, terrain: Terrain, nodes: list[Node],
                 channels: list[Channel], max_link_m: float = 900.0,
                 max_path_loss_db: float = 118.0):
        self.terrain = terrain
        self.nodes = {n.id: n for n in nodes}
        self.channels = channels
        self.links: dict[int, Link] = {}
        self.adj: dict[int, list[int]] = {n.id: [] for n in nodes}  # node -> link ids
        lid = 0
        ids = list(self.nodes)
        for i, u in enumerate(ids):
            for v in ids[i + 1:]:
                a, b = self.nodes[u], self.nodes[v]
                d = math.dist((a.x, a.y), (b.x, b.y))
                if d > max_link_m:
                    continue
                pl = self.path_loss_db(a, b)
                if pl > max_path_loss_db:
                    continue
                lk = Link(lid, u, v, d, pl)
                self.links[lid] = lk
                self.adj[u].append(lid)
                self.adj[v].append(lid)
                lid += 1

    def path_loss_db(self, a: Node, b: Node) -> float:
        """对数距离衰减(2.4G 近地 n≈2.7) + 地形遮挡附加损耗."""
        d = max(1.0, math.dist((a.x, a.y), (b.x, b.y)))
        pl = 40.0 + 10 * 2.7 * math.log10(d) + self.terrain.obstruction_loss_db(a, b)
        return pl

    def rx_power_dbm(self, from_node: int, to_node: int) -> float:
        a, b = self.nodes[from_node], self.nodes[to_node]
        return P_TX_DBM + 2 * ANTENNA_GAIN_DB - self.path_loss_db(a, b)

    # ---- 干扰 / SINR(物理干扰模型,干扰累加) ----
    def sinr_db(self, rx_node: int, tx_node: int, chan: Channel,
                interferers: list[tuple[int, Channel]]) -> float:
        """rx_node 收 tx_node,interferers 为同时发射的 (发射节点, 信道) 列表."""
        sig = 10 ** (self.rx_power_dbm(tx_node, rx_node) / 10)
        interf = 10 ** (NOISE_DBM / 10)
        for w, cw in interferers:
            if w == tx_node:
                continue
            f = overlap_factor(chan, cw)
            if f <= 0:
                continue
            interf += f * 10 ** (self.rx_power_dbm(w, rx_node) / 10)
        return 10 * math.log10(sig / interf)

    def rate_mbps(self, sinr_db: float) -> float:
        if sinr_db < SINR_THRESHOLD_DB:
            return 0.0
        return min(RATE_CAP_MBPS, BW_MHZ * math.log2(1 + 10 ** (sinr_db / 10)))

    def clean_rate_mbps(self, link_id: int, chan: Channel) -> float:
        lk = self.links[link_id]
        return self.rate_mbps(self.sinr_db(lk.b, lk.a, chan, []))

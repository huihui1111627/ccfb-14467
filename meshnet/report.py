"""生成 HTML 报告:地形图(canvas 渲染高程)、链路按信道上色、时隙表、指标对比."""
from __future__ import annotations

import json

from .model import Network
from .planner import Plan

# 信道 -> 颜色
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
           "#f032e6", "#bfef45", "#469990", "#9A6324", "#800000", "#000075",
           "#a9a9a9", "#e6beff", "#ffd8b1"]


def _ch_color(idx, channels):
    order = sorted({c.idx for c in channels})
    return PALETTE[order.index(idx) % len(PALETTE)] if idx in order else "#888"


def render_html(net: Network, plans: dict, sims: dict, main: str,
                dfs_info: dict, exhaustive_note: str,
                dfs_pre: float = 0.0, dfs_post: float = 0.0) -> str:
    """plans/sims: {方案名: (Plan, SimResult)}; main: 主方案名."""
    terr = net.terrain
    plan = plans[main]
    sim = sims[main]

    data = {
        "terrain": {"elev": [[round(v, 1) for v in row] for row in terr.elev.tolist()],
                    "size": terr.size_m},
        "nodes": [{"id": n.id, "x": n.x, "y": n.y, "gw": n.gateway,
                   "radios": plan.radio_channels[n.id],
                   "demand": n.demand_mbps} for n in net.nodes.values()],
        "links": [{"id": lk.id, "a": lk.a, "b": lk.b,
                   "ch": plan.link_channel.get(lk.id),
                   "load": None} for lk in net.links.values()],
        "channels": [{"idx": c.idx, "freq": c.freq_mhz, "dfs": c.dfs,
                      "color": _ch_color(c.idx, plan.channels)}
                     for c in plan.channels],
        "frame_slots": len(plan.slot_links),
        "slot_links": plan.slot_links,
        "link_slots": {str(k): v for k, v in plan.link_slots.items()},
        "routes": {str(k): v for k, v in plan.routes.items()},
        "backup": {str(n): {str(r): c for r, c in d.items()}
                   for n, d in plan.backup_channels.items()},
    }

    rows_nodes = ""
    for nid, r in sorted(sim.per_node.items()):
        rows_nodes += (f"<tr><td>杆 {nid}</td><td>{r.hops}</td>"
                       f"<td>{r.throughput_mbps:.2f}</td><td>{r.delay_ms:.1f}</td></tr>")

    rows_cmp = ""
    for name, s in sims.items():
        worst = s.per_node[s.worst_node]
        avg_t = sum(r.throughput_mbps for r in s.per_node.values()) / max(1, len(s.per_node))
        avg_d = sum(r.delay_ms for r in s.per_node.values()) / max(1, len(s.per_node))
        rows_cmp += (f"<tr><td>{name}</td><td>杆 {s.worst_node}</td>"
                     f"<td>{s.worst_throughput:.2f}</td><td>{avg_t:.2f}</td>"
                     f"<td>{worst.delay_ms:.1f}</td><td>{avg_d:.1f}</td></tr>")

    dfs_rows = ""
    for ev in dfs_info.get("events", []):
        dfs_rows += (f"<tr><td>第 {ev['frame']} 帧</td>"
                     f"<td>{ev['hit_channels']}</td><td>{ev['switched_radios']}</td></tr>")

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>山区景区无线回传 · 信道/时隙规划报告</title>
<style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;margin:24px;background:#fafafa;color:#222}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;font-size:13px;background:#fff}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:center}}
th{{background:#eee}}
canvas{{background:#fff;border:1px solid #ccc}}
.flex{{display:flex;gap:24px;flex-wrap:wrap}}
.slotcell{{width:16px;height:16px;display:inline-block;margin:0}}
.note{{color:#666;font-size:12px}}
</style></head><body>
<h1>多射频网状网信道分配与链路调度 — 规划报告</h1>
<div class="flex">
<div><h2>地形与链路(按信道上色,方块=有线出口,圆=杆)</h2>
<canvas id="map" width="640" height="640"></canvas>
<div id="legend" class="note"></div></div>
<div><h2>时隙表(行=链路,列=时隙,颜色=信道)</h2>
<canvas id="slots" width="640" height="360"></canvas>
<div class="note">顺流调度:离出口远的链路排在靠前的时隙,包在一个周期内逐级汇入。</div></div>
</div>

<h2>各杆端到端指标(主方案:{main})</h2>
<table><tr><th>节点</th><th>跳数</th><th>吞吐 Mbps</th><th>时延 ms</th></tr>
{rows_nodes}</table>
<p><b>最差杆子:杆 {sim.worst_node},吞吐 {sim.worst_throughput:.2f} Mbps</b></p>

<h2>方案对比(同一验收仿真:逐时隙 SINR 判定)</h2>
<table><tr><th>方案</th><th>最差节点</th><th>最差吞吐 Mbps</th><th>平均吞吐 Mbps</th>
<th>最差时延 ms</th><th>平均时延 ms</th></tr>{rows_cmp}</table>
<p class="note">{exhaustive_note}</p>

<h2>DFS 雷达事件</h2>
<table><tr><th>触发时刻</th><th>被命中信道</th><th>切换射频数</th></tr>{dfs_rows}</table>
<p>命中后 DFS 链路立即切到预选备用信道(切换前已在各自时隙做过 SINR 预演),
断链时隙 {dfs_info.get("outage_slots", 0)} 个;
最差杆吞吐 {dfs_pre:.2f} → {dfs_post:.2f} Mbps,业务不中断、降级运行。</p>

<script>
const D = {json.dumps(data)};
const W = 640, H = 640;
// ---------- 地形 ----------
const cv = document.getElementById('map'), ctx = cv.getContext('2d');
const elev = D.terrain.elev, n = elev.length, S = D.terrain.size;
let mn = 1e9, mx = -1e9;
for (const row of elev) for (const v of row) {{ mn = Math.min(mn,v); mx = Math.max(mx,v); }}
const img = ctx.createImageData(W, H);
for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {{
  const gx = px/W*(n-1), gy = py/H*(n-1);
  const x0 = Math.floor(gx), y0 = Math.floor(gy);
  const v = elev[Math.min(y0,n-1)][Math.min(x0,n-1)];
  const t = (v-mn)/(mx-mn+1e-9);
  const i = (py*W+px)*4;
  img.data[i]   = 230 - 120*t;
  img.data[i+1] = 225 - 100*t;
  img.data[i+2] = 210 - 130*t;
  img.data[i+3] = 255;
}}
ctx.putImageData(img, 0, 0);
const X = m => m/S*W, Y = m => m/S*H;
const chColor = {{}};
for (const c of D.channels) chColor[c.idx] = c.color;
// 链路
for (const lk of D.links) {{
  const a = D.nodes.find(n=>n.id===lk.a), b = D.nodes.find(n=>n.id===lk.b);
  ctx.strokeStyle = lk.ch==null ? '#bbb' : chColor[lk.ch];
  ctx.lineWidth = lk.ch==null ? 1 : 3;
  ctx.beginPath(); ctx.moveTo(X(a.x), Y(a.y)); ctx.lineTo(X(b.x), Y(b.y)); ctx.stroke();
}}
// 节点
for (const nd of D.nodes) {{
  ctx.fillStyle = nd.gw ? '#111' : '#fff';
  ctx.strokeStyle = '#111'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  if (nd.gw) ctx.rect(X(nd.x)-6, Y(nd.y)-6, 12, 12);
  else ctx.arc(X(nd.x), Y(nd.y), 6, 0, 7);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#111'; ctx.font = '11px sans-serif';
  ctx.fillText((nd.gw?'出口':'杆')+nd.id, X(nd.x)+8, Y(nd.y)+4);
}}
// 图例
let lg = '信道: ';
for (const c of D.channels) lg += `<span style="color:${{c.color}}">■</span>${{c.idx}}${{c.dfs?'(DFS)':''}} `;
document.getElementById('legend').innerHTML = lg;

// ---------- 时隙表 ----------
const cv2 = document.getElementById('slots'), c2 = cv2.getContext('2d');
const T = D.frame_slots;
const usedLinks = Object.keys(D.link_slots).map(Number).sort((a,b)=>a-b);
const cw = (W-60)/T, rh = Math.min(20, (H-30)/Math.max(1,usedLinks.length));
c2.font = '10px sans-serif';
usedLinks.forEach((lid, i) => {{
  const y = 20 + i*rh;
  c2.fillStyle = '#333'; c2.fillText('L'+lid, 4, y+rh*0.75);
  const lk = D.links.find(l=>l.id===lid);
  for (const s of D.link_slots[lid]) {{
    c2.fillStyle = chColor[lk.ch] || '#888';
    c2.fillRect(50+s*cw, y, cw-1, rh-2);
  }}
}});
for (let s = 0; s < T; s += 4) {{ c2.fillStyle='#999'; c2.fillText(s, 50+s*cw, 12); }}
</script>
</body></html>"""

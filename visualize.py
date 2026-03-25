from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from networkx.algorithms import community
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import io, re, json

# ── 中文字体自动检测（Windows / Linux / macOS 均兼容）──
def _find_chinese_font() -> str | None:
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "STHeiti",  # Windows / macOS
        "WenQuanYi Micro Hei", "Noto Sans CJK SC",          # Linux
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

_CN_FONT = _find_chinese_font()
if _CN_FONT:
    matplotlib.rcParams["font.family"] = _CN_FONT
matplotlib.rcParams["axes.unicode_minus"] = False  # 负号不乱码

router = APIRouter(prefix="/visualization", tags=["visualization"])


# ══════════════════════════════════════════════════════
#  文件解析
# ══════════════════════════════════════════════════════

def parse_pajek(content: str) -> nx.DiGraph:
    """支持 *Vertices / *Arcs (有向带权) / *Edges (无向带权)"""
    G = nx.DiGraph()
    section = None

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        lower = line.lower()
        if lower.startswith("*vertices"):
            section = "vertices"; continue
        elif lower.startswith("*arcs"):
            section = "arcs"; continue
        elif lower.startswith("*edges"):
            section = "edges"; continue

        if section == "vertices":
            m = re.match(r'^(\d+)\s+"(.+)"', line)
            if m:
                G.add_node(int(m.group(1)), label=m.group(2))

        elif section in ("arcs", "edges"):
            parts = line.split()
            if len(parts) >= 2:
                src, tgt = int(parts[0]), int(parts[1])
                weight = float(parts[2]) if len(parts) >= 3 else 1.0
                G.add_edge(src, tgt, weight=weight)
                if section == "edges":
                    G.add_edge(tgt, src, weight=weight)
    return G


def parse_network_file(content: str, filename: str) -> nx.DiGraph:
    """根据文件扩展名选择解析方式"""
    ext = filename.rsplit(".", 1)[-1].lower()
    try:
        if ext in ("net", "pajek"):
            return parse_pajek(content)
        if ext == "gml":
            return nx.read_gml(io.StringIO(content))
        if ext == "graphml":
            return nx.read_graphml(io.StringIO(content))
        if ext in ("txt", "edgelist"):
            return nx.read_edgelist(io.StringIO(content))
        if ext == "json":
            return nx.node_link_graph(json.loads(content))
    except Exception:
        pass
    # 兜底：尝试 Pajek
    return parse_pajek(content)


async def _parse_upload(file: UploadFile) -> nx.DiGraph:
    """读取上传文件并解析为 NetworkX 图，供各接口复用"""
    raw = await file.read()
    content = raw.decode("utf-8", errors="ignore")
    try:
        G = parse_network_file(content, file.filename or "network.net")
    except Exception as e:
        raise HTTPException(400, f"文件解析失败: {e}")
    if G.number_of_nodes() == 0:
        raise HTTPException(400, "网络文件中未找到节点数据，请检查文件格式")
    return G


# ══════════════════════════════════════════════════════
#  绘图
# ══════════════════════════════════════════════════════

LAYOUT_MAP = {
    "force":        lambda G: nx.spring_layout(G, seed=42, k=1.8 / max(len(G) ** 0.5, 1)),
    "spring":       lambda G: nx.spring_layout(G, seed=42),
    "circular":     nx.circular_layout,
    "kamada_kawai": nx.kamada_kawai_layout,
    "spectral":     nx.spectral_layout,
    "shell":        nx.shell_layout,
}

THEMES = {
    "light": {"bg": "#f8fbff", "panel": "#ffffff", "edge": "#b0c4de", "text": "#1f2a37", "cmap": cm.viridis},
    "dark":  {"bg": "#0f172a", "panel": "#1e293b", "edge": "#334155", "text": "#e2e8f0", "cmap": cm.plasma},
}


def draw_network(G: nx.DiGraph, layout: str, theme: str, description: str) -> io.BytesIO:
    # 超过 300 节点取度最高的子图
    if G.number_of_nodes() > 300:
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:300]
        G = G.subgraph([n for n, _ in top_nodes]).copy()

    th = THEMES.get(theme, THEMES["dark"])
    pos = LAYOUT_MAP.get(layout, LAYOUT_MAP["force"])(G)

    degree  = dict(G.degree())
    max_deg = max(degree.values(), default=1)
    node_sizes  = [80 + 600 * (degree[n] / max_deg) ** 0.7 for n in G.nodes]
    node_colors = [degree[n] / max_deg for n in G.nodes]

    weights    = [G.edges[u, v].get("weight", 1.0) for u, v in G.edges]
    max_w      = max(weights, default=1)
    edge_widths = [0.3 + 2.2 * (w / max_w) for w in weights]
    edge_alphas = [0.25 + 0.55 * (w / max_w) for w in weights]

    # 备注含"社团"/"community"时启用社团着色
    use_community = "社团" in description or "community" in description.lower()
    if use_community:
        G_und  = G.to_undirected()
        comms  = list(community.greedy_modularity_communities(G_und))
        comm_map = {n: i for i, c in enumerate(comms) for n in c}
        n_comms  = max(comm_map.values(), default=0) + 1
        palette  = plt.cm.get_cmap("tab20", n_comms)
        node_colors = [palette(comm_map.get(n, 0) / n_comms) for n in G.nodes]
        cmap_arg = None
    else:
        cmap_arg = th["cmap"]

    fig, ax = plt.subplots(figsize=(14, 11), facecolor=th["bg"])
    ax.set_facecolor(th["bg"])

    for (u, v), width, alpha in zip(G.edges(), edge_widths, edge_alphas):
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=th["edge"], linewidth=width, alpha=alpha, zorder=1)

    sc_kwargs = dict(
        x=[pos[n][0] for n in G.nodes],
        y=[pos[n][1] for n in G.nodes],
        s=node_sizes, c=node_colors,
        alpha=0.88, zorder=2, linewidths=0.6, edgecolors=th["panel"]
    )
    if cmap_arg:
        scatter = ax.scatter(**sc_kwargs, cmap=cmap_arg)
        cb = plt.colorbar(scatter, ax=ax, fraction=0.025, pad=0.01, label="归一化度")
        cb.ax.yaxis.label.set_color(th["text"])
        cb.ax.tick_params(colors=th["text"])
    else:
        ax.scatter(**sc_kwargs)

    # 只给度最高的 40 个节点加标签
    label_threshold = sorted(degree.values(), reverse=True)[min(40, len(degree) - 1)]
    for n in G.nodes:
        if degree[n] >= label_threshold:
            lbl = G.nodes[n].get("label", str(n))[-18:]
            ax.text(pos[n][0], pos[n][1], lbl,
                    fontsize=6.5, color=th["text"], ha="center", va="center",
                    fontweight="bold", zorder=3)

    info = (f"节点: {G.number_of_nodes()}  "
            f"边: {G.number_of_edges()}  "
            f"密度: {nx.density(G):.4f}  "
            f"布局: {layout}")
    ax.text(0.01, 0.01, info, transform=ax.transAxes,
            fontsize=8, color=th["text"], alpha=0.65, va="bottom")

    if description:
        ax.set_title(description, color=th["text"], fontsize=11, pad=10)

    ax.axis("off")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=th["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════
#  接口
# ══════════════════════════════════════════════════════

# ── POST /visualization/render ────────────────────────
@router.post("/render")
async def render(
    file: UploadFile = File(...),
    user_id: str = Form(default=""),
    options: str = Form(default="{}"),
):
    try:
        opts = json.loads(options)
    except Exception:
        opts = {}

    layout      = opts.get("layout", "force")
    theme       = opts.get("theme", "dark")
    description = opts.get("description", "")

    G   = await _parse_upload(file)
    buf = draw_network(G, layout, theme, description)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=network.png"},
    )


# ── POST /visualization/stats ─────────────────────────
@router.post("/stats")
async def stats_endpoint(file: UploadFile = File(...)):
    G     = await _parse_upload(file)
    G_und = G.to_undirected()
    deg   = dict(G.degree())
    return {
        "num_nodes":   G.number_of_nodes(),
        "num_edges":   G.number_of_edges(),
        "density":     round(nx.density(G), 6),
        "avg_degree":  round(sum(deg.values()) / len(deg), 3) if deg else 0,
        "avg_clustering": round(nx.average_clustering(G_und), 4),
        "is_weakly_connected": nx.is_weakly_connected(G),
        "num_weakly_connected_components": nx.number_weakly_connected_components(G),
    }


# ── POST /visualization/centrality ───────────────────
@router.post("/centrality")
async def centrality_endpoint(file: UploadFile = File(...), top: int = 15):
    G = await _parse_upload(file)

    def top_nodes(d):
        return [
            {"node": str(k), "label": G.nodes[k].get("label", str(k)), "value": round(v, 6)}
            for k, v in sorted(d.items(), key=lambda x: -x[1])[:top]
        ]

    # pagerank: 优先用 scipy 版，没装 scipy 则用纯 Python 版
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = nx.pagerank_numpy(G, weight="weight")

    return {
        "degree":      top_nodes(nx.degree_centrality(G)),
        "betweenness": top_nodes(nx.betweenness_centrality(G, weight="weight")),
        "closeness":   top_nodes(nx.closeness_centrality(G)),
        "pagerank":    top_nodes(pr),
    }


# ── POST /visualization/communities ──────────────────
@router.post("/communities")
async def communities_endpoint(file: UploadFile = File(...)):
    G     = await _parse_upload(file)
    G_und = G.to_undirected()
    comms = list(community.greedy_modularity_communities(G_und))
    mod   = community.modularity(G_und, comms)
    return {
        "num_communities": len(comms),
        "modularity":      round(mod, 4),
        "communities": [
            {
                "id":    i,
                "size":  len(c),
                "nodes": [{"id": str(n), "label": G.nodes[n].get("label", str(n))} for n in c],
            }
            for i, c in enumerate(comms)
        ],
    }
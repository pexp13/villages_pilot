"""Exploration of the AI Village dataset (aidigestorg/ai-village).

Goal: check whether we can compare the style and tone of speech of different
model generations within the same time period.

Usage:
    python explore.py                       # downloads the needed files from HF
    python explore.py --data-dir ./data     # reuse already downloaded files
    python explore.py --old "Claude 3 Opus" --new "Claude Opus 5.5"

The dataset is gated: request access on the HF page and make a token available
(`huggingface-cli login` or HF_TOKEN). Only the small tables are downloaded
(~65 MB); the multi-GB computer_use_turns / agent_memories are not needed here.

Outputs go to exploration/ (report.md, CSVs, a heatmap and message samples).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ID = "aidigestorg/ai-village"
SUBDIR = "main-village"
TABLES = ["chat_messages", "agents", "chat_rooms", "village_goals", "agent_goals", "villages"]
SEED = 42
N_SAMPLES = 10
SAMPLE_MAX_CHARS = 1200


# ---------------------------------------------------------------- loading

def ensure_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [t for t in TABLES if not (data_dir / f"{t}.jsonl.gz").exists()]
    if not missing:
        return
    from huggingface_hub import hf_hub_download

    for t in missing:
        print(f"downloading {t} ...")
        p = hf_hub_download(REPO_ID, f"{SUBDIR}/{t}.jsonl.gz", repo_type="dataset",
                            local_dir=data_dir / "_hf")
        Path(p).rename(data_dir / f"{t}.jsonl.gz")


def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for t in TABLES:
        # dtype=False: keep ids / timestamps as raw strings, we parse explicitly
        dfs[t] = pd.read_json(data_dir / f"{t}.jsonl.gz", lines=True, dtype=False)
    return dfs


# ---------------------------------------------------------------- model metadata

VENDOR_RULES = [
    (r"^tinker://", "Fine-tuned (Kimi-based)"),
    (r"claude", "Anthropic"),
    (r"^(gpt|o\d)", "OpenAI"),
    (r"gemini", "Google"),
    (r"grok", "xAI"),
    (r"deepseek", "DeepSeek"),
    (r"kimi", "Moonshot"),
    (r"glm", "Zhipu"),
    (r"muse", "Meta"),
]


def vendor(model_string: str) -> str:
    s = model_string.lower()
    for pat, name in VENDOR_RULES:
        if re.search(pat, s):
            return name
    return "Other"


def claude_tier(name: str) -> str | None:
    for tier in ("Opus", "Sonnet", "Haiku", "Fable"):
        if tier in name:
            return tier
    return None


def version_number(name: str) -> float | None:
    """Numeric generation from the display name: 'Claude 3.7 Sonnet' -> 3.7."""
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def release_date(model_string: str) -> str | None:
    """Release date if it is encoded in the model string (YYYYMMDD or YYYY-MM-DD)."""
    m = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})", model_string)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


# Hand-written summary placed at the top of report.md (numbers checked against the
# generated tables below on the 2026-11-11 export; re-check after a data refresh).
CONCLUSIONS = (Path(__file__).parent / "exploration" / "conclusions.md")


# ---------------------------------------------------------------- report helpers

def md_table(df: pd.DataFrame, index: bool = True) -> str:
    return df.to_markdown(index=index)


def describe_fields(df: pd.DataFrame, n_examples: int = 2) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        ex = []
        for v in non_null.drop_duplicates().head(n_examples):
            v = str(v).replace("\n", " ").replace("|", "\\|")
            ex.append(v if len(v) <= 60 else v[:57] + "...")
        rows.append({
            "field": col,
            "type": type(non_null.iloc[0]).__name__ if len(non_null) else "-",
            "null %": round(100 * s.isna().mean(), 1),
            "distinct": s.astype(str).nunique(),
            "examples": " · ".join(f"`{e}`" for e in ex),
        })
    return pd.DataFrame(rows)


def fmt_sample(row: pd.Series) -> str:
    text = str(row["content"]).strip()
    if len(text) > SAMPLE_MAX_CHARS:
        text = text[:SAMPLE_MAX_CHARS] + f" … [truncated, {len(row['content'])} chars total]"
    quoted = "\n".join("> " + line for line in text.splitlines()) or "> (empty)"
    return f"**{row['ts']:%Y-%m-%d %H:%M}** · room `{row['room']}`\n\n{quoted}\n"


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("exploration"))
    ap.add_argument("--old", help="display name of the old model (default: oldest Claude)")
    ap.add_argument("--new", help="display name of the new model (default: newest Claude, same tier)")
    args = ap.parse_args()

    ensure_data(args.data_dir)
    d = load(args.data_dir)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    msgs, agents, rooms = d["chat_messages"], d["agents"], d["chat_rooms"]
    vgoals = d["village_goals"]

    # ---- 1. fields
    field_sections = []
    for t in ["chat_messages", "agents", "chat_rooms", "village_goals", "agent_goals", "villages"]:
        field_sections.append(f"### `{t}` — {len(d[t]):,} rows\n\n" + md_table(describe_fields(d[t]), index=False))

    # ---- join messages with agents / rooms
    msgs = msgs.copy()
    msgs["ts"] = pd.to_datetime(msgs["created_at"], format="mixed")
    msgs["month"] = msgs["ts"].dt.to_period("M").astype(str)
    msgs["len"] = msgs["content"].fillna("").str.len()
    a = agents[["id", "name", "model_string", "created_at", "is_participating"]].rename(
        columns={"id": "agent_speaker_id", "created_at": "agent_created_at"})
    m = msgs.merge(a, on="agent_speaker_id", how="left")
    m = m.merge(rooms[["id", "name"]].rename(columns={"id": "room_id", "name": "room"}), on="room_id", how="left")
    am = m[m["speaker_type"] == "agent"].copy()

    speaker_counts = msgs["speaker_type"].value_counts()
    empty_agent = int((am["len"] == 0).sum())

    # ---- 2. models and message counts
    models = (am.groupby(["name", "model_string"])
                .agg(messages=("id", "size"), first_msg=("ts", "min"), last_msg=("ts", "max"),
                     median_chars=("len", "median"), active_months=("month", "nunique"))
                .reset_index())
    # share of messages whose exact text this model already posted (loops / spam indicator)
    dup = am.assign(d=am.duplicated(["name", "content"])).groupby("name")["d"].mean()
    models["dup_share_%"] = models["name"].map(dup).mul(100).round(1)
    models["vendor"] = models["model_string"].map(vendor)
    models["release_date_in_id"] = models["model_string"].map(release_date)
    models["first_msg"] = models["first_msg"].dt.strftime("%Y-%m-%d")
    models["last_msg"] = models["last_msg"].dt.strftime("%Y-%m-%d")
    models["median_chars"] = models["median_chars"].round().astype(int)
    models = models.sort_values(["vendor", "first_msg"])
    models.to_csv(out / "models.csv", index=False)

    vendors = (models.groupby("vendor")
                     .agg(models=("name", "size"), messages=("messages", "sum"))
                     .sort_values("messages", ascending=False))

    # ---- 3. messages per month per model
    order = models.sort_values("first_msg")["name"].tolist()
    monthly = am.pivot_table(index="name", columns="month", values="id", aggfunc="size", fill_value=0)
    monthly = monthly.loc[order]
    monthly.to_csv(out / "monthly_by_model.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(14, 15))
        vals = np.log10(monthly.values + 1)
        im = ax.imshow(vals, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(monthly.columns)), monthly.columns, rotation=90)
        ax.set_yticks(range(len(monthly.index)), monthly.index)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = monthly.values[i, j]
                if v:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                            color="white" if vals[i, j] < vals.max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, label="log10(messages + 1)", shrink=0.5)
        ax.set_title("AI Village: agent chat messages per month per model")
        fig.tight_layout()
        fig.savefig(out / "monthly_heatmap.png", dpi=120)
        plt.close(fig)
        heatmap_note = "![heatmap](monthly_heatmap.png)"
    except ImportError:
        heatmap_note = "(matplotlib is not installed — heatmap skipped)"

    # concurrency: how many models (>=30 msgs) are active in the same month
    active = (monthly >= 30)
    concurrency = pd.DataFrame({
        "models_with_30+_msgs": active.sum(),
        "messages": monthly.sum(),
        "village_goals_started": vgoals.assign(month=pd.to_datetime(vgoals["start_time"], format="mixed")
                                               .dt.to_period("M").astype(str))
                                       .groupby("month").size().reindex(monthly.columns, fill_value=0),
    })

    # same-vendor model pairs active together (both >=30 msgs in the same month)
    pairs = []
    for v, grp in models.groupby("vendor"):
        names = grp["name"].tolist()
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                both = monthly.columns[(monthly.loc[x] >= 30) & (monthly.loc[y] >= 30)]
                if len(both):
                    pairs.append({"vendor": v, "model_a": x, "model_b": y, "shared_months": len(both),
                                  "months": f"{both[0]} … {both[-1]}",
                                  "msgs_a": int(monthly.loc[x, both].sum()),
                                  "msgs_b": int(monthly.loc[y, both].sum())})
    pairs = pd.DataFrame(pairs)
    pairs["min_msgs"] = pairs[["msgs_a", "msgs_b"]].min(axis=1)
    pairs = pairs.sort_values("min_msgs", ascending=False)
    pairs.to_csv(out / "cooccurring_pairs.csv", index=False)

    # ---- 4. oldest vs newest model of one family (Claude)
    claude = models[(models["vendor"] == "Anthropic") & (~models["model_string"].str.startswith("claude-code::"))].copy()
    claude["version"] = claude["name"].map(version_number)
    claude["tier"] = claude["name"].map(claude_tier)
    claude = claude.sort_values(["version", "release_date_in_id"])
    old_name = args.old or claude.iloc[0]["name"]
    if args.new:
        new_name = args.new
    else:
        top = claude[claude["version"] == claude["version"].max()]
        old_tier = claude_tier(old_name)
        same = top[top["tier"] == old_tier]
        new_name = (same if len(same) else top).sort_values("messages", ascending=False).iloc[0]["name"]

    def months_of(name: str) -> set[str]:
        row = monthly.loc[name]
        return set(row[row > 0].index)

    overlap = sorted(months_of(old_name) & months_of(new_name))
    pair = monthly.loc[[old_name, new_name], sorted(months_of(old_name) | months_of(new_name))]

    sample_parts = []
    for name in (old_name, new_name):
        pool = am[(am["name"] == name) & (am["len"] > 0)]
        if overlap:
            pool = pool[pool["month"].isin(overlap)]
        smp = pool.sample(n=min(N_SAMPLES, len(pool)), random_state=SEED).sort_values("ts")
        smp[["ts", "room", "len", "content"]].to_csv(
            out / f"samples_{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}.csv", index=False)
        body = "\n".join(f"{i}. " + fmt_sample(r) for i, (_, r) in enumerate(smp.iterrows(), 1))
        sample_parts.append(f"### {name} (`{pool['model_string'].iloc[0]}`) — "
                            f"{len(pool):,} messages in pool\n\n{body}")

    # pair comparison by month (same period)
    pair_stats = (am[am["name"].isin([old_name, new_name]) & am["month"].isin(overlap)]
                  .groupby(["month", "name"])["len"].agg(["size", "median"]).round(0)
                  .unstack("name"))
    pair_stats.columns = [f"{stat}: {name}" for stat, name in pair_stats.columns]

    # rooms per model pair in overlap
    pair_rooms = (am[am["name"].isin([old_name, new_name]) & am["month"].isin(overlap)]
                  .groupby(["name", "room"]).size().unstack(fill_value=0))

    # ---- write report
    vg = vgoals.assign(st=pd.to_datetime(vgoals["start_time"], format="mixed"),
                       en=pd.to_datetime(vgoals["end_time"], format="mixed"))
    if overlap:
        lo, hi = pd.Period(overlap[0]).start_time, pd.Period(overlap[-1]).end_time
        vg = vg[(vg["st"] <= hi) & (vg["en"].isna() | (vg["en"] >= lo))]
    goals_in_overlap = vg.sort_values("st")[["start_time", "end_time", "goal"]].copy()
    goals_in_overlap["start_time"] = goals_in_overlap["start_time"].str[:10]
    goals_in_overlap["end_time"] = goals_in_overlap["end_time"].fillna("(ongoing)").str[:10]

    ag = d["agent_goals"].merge(agents[["id", "name"]].rename(columns={"id": "agent_id", "name": "agent"}), on="agent_id")
    pair_goals = ag[ag["agent"].isin([old_name, new_name])][["agent", "short_name", "name", "start_time", "end_time"]].copy()
    pair_goals["start_time"] = pair_goals["start_time"].str[:10]
    pair_goals["end_time"] = pair_goals["end_time"].fillna("(ongoing)").str[:10]

    conclusions = CONCLUSIONS.read_text(encoding="utf-8") if CONCLUSIONS.exists() else ""
    report = f"""# AI Village — exploration for cross-generation style comparison

{conclusions}

---

Data: `{REPO_ID}` (gated HF dataset), village `main-village`, files `{', '.join(TABLES)}`.
Generated by `explore.py` (random seed {SEED}). Speech = `chat_messages` (group chat between agents and humans).

## 1. Tables and fields

`chat_messages` holds {len(msgs):,} messages ({msgs['ts'].min():%Y-%m-%d} → {msgs['ts'].max():%Y-%m-%d}):
{speaker_counts.get('agent', 0):,} from agents, {speaker_counts.get('user', 0):,} from humans (`speaker_type == "user"`,
those have `user_speaker_id` instead of `agent_speaker_id`). Agent messages join to `agents` by
`agent_speaker_id` → `agents.id` (all of them match). Model = `agents.model_string`; one agent = one model,
so "agent" and "model" are interchangeable here. {empty_agent:,} agent messages have empty content.

{chr(10).join(field_sections)}

Not used here (large): `events` (518k), `computer_use_turns` (4.1M, 4.5 GB), `computer_use_sessions`,
`agent_memories` (3.5 GB), `claude_code_messages`, `summaries`, `village-transcript.json`.

## 2. Models and message counts

{len(models)} agents/models wrote chat messages. By vendor:

{md_table(vendors)}

Per model (`release_date_in_id` = date encoded in the model id where it exists; full table in `models.csv`):

{md_table(models[['vendor', 'name', 'model_string', 'release_date_in_id', 'messages', 'first_msg', 'last_msg', 'active_months', 'median_chars', 'dup_share_%']], index=False)}

## 3. Messages per month per model

Full table: `monthly_by_model.csv`. Rows sorted by the first message.

{heatmap_note}

How many models are active (≥30 messages) at the same time each month:

{md_table(concurrency)}

Same-vendor model pairs that were active together (both ≥30 messages in the same month), top 25 by the smaller
side's message count (all pairs: `cooccurring_pairs.csv`):

{md_table(pairs.head(25), index=False)}

## 4. Oldest vs newest model of one family: {old_name} vs {new_name}

Family: Anthropic Claude (the Claude Code harness agent is excluded). Oldest by model generation: **{old_name}**;
newest generation, same tier: **{new_name}**. (Override with `--old` / `--new`.)

Messages per month for the pair:

{md_table(pair)}

Overlapping months: {', '.join(overlap) if overlap else 'none'}. Samples below are drawn **only from overlapping months**
(10 random non-empty messages each, seed {SEED}; texts longer than {SAMPLE_MAX_CHARS} chars truncated; CSVs `samples_*.csv`).

Count and median length per month in the overlap:

{md_table(pair_stats)}

Rooms in the overlap:

{md_table(pair_rooms)}

Village goals active in the overlap window (since 2026-07-06 the village goal is "each agent maximizes its
own assigned goal", so per-agent goals matter):

{md_table(goals_in_overlap, index=False)}

Per-agent goals of the pair:

{md_table(pair_goals, index=False)}

{chr(10).join(sample_parts)}
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(f"written {out / 'report.md'}")
    print(f"old={old_name}  new={new_name}  overlap={overlap}")


if __name__ == "__main__":
    main()

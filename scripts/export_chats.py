"""
export_chats.py — Convert Claude Code JSONL conversation logs to readable .txt files.

Usage:
    python scripts/export_chats.py [PROJECT_DIR] [OUTPUT_DIR]

Arguments:
    PROJECT_DIR  Path to the Claude Code project directory containing .jsonl files.
                 Default: ~/.claude/projects/<encoded-repo-name>
                 The encoded name is the repo's absolute path with slashes replaced
                 by hyphens (e.g. /home/user/code/myrepo → c--home-user-code-myrepo).
    OUTPUT_DIR   Directory to write .txt transcripts to.
                 Default: output/chats/ relative to this script's repo root.

Output: one .txt file per conversation in OUTPUT_DIR/
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]

if len(sys.argv) >= 2:
    PROJECT_DIR = Path(sys.argv[1])
else:
    # Claude Code stores project logs under ~/.claude/projects/<encoded-path>
    # where the encoded path replaces path separators with hyphens.
    # You may need to adjust this to match your local project directory name.
    encoded = ROOT.as_posix().lstrip("/").replace("/", "-").replace(":", "")
    PROJECT_DIR = Path.home() / ".claude" / "projects" / encoded

if len(sys.argv) >= 3:
    OUTPUT_DIR = Path(sys.argv[2])
else:
    OUTPUT_DIR = ROOT / "output" / "chats"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_ts(ts_str: str) -> str:
    """Convert ISO timestamp to a human-readable local string."""
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        local = dt.astimezone()
        return local.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_str[:19]


def extract_text_from_content(content) -> str:
    """
    Extract human-readable text from a message content field.
    Content can be a plain string or a list of typed blocks.
    """
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(text)

        elif btype == "tool_use":
            tool_name = block.get("name", "unknown_tool")
            tool_input = block.get("input", {})
            # Show tool name and a brief summary of inputs
            summary = _summarize_tool_input(tool_name, tool_input)
            parts.append(f"[Tool: {tool_name}]{summary}")

        elif btype == "tool_result":
            # Tool results are voluminous; include a short excerpt
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                result_text = " ".join(
                    b.get("text", "") for b in result_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                result_text = str(result_content)
            excerpt = result_text[:300].strip()
            if len(result_text) > 300:
                excerpt += f"... [{len(result_text)} chars total]"
            if excerpt:
                parts.append(f"[Tool result: {excerpt}]")

    return "\n".join(parts)


def _summarize_tool_input(tool_name: str, tool_input: dict) -> str:
    """Return a one-line summary of a tool call's key inputs."""
    if not tool_input:
        return ""

    if tool_name in ("Read", "Write", "Edit"):
        fp = tool_input.get("file_path", "")
        if fp:
            return f" → {Path(fp).name}"

    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        return f" → {cmd[:120]}" if cmd else ""

    if tool_name == "Grep":
        pat = tool_input.get("pattern", "")
        return f" → pattern: {pat[:80]}" if pat else ""

    if tool_name == "Glob":
        pat = tool_input.get("pattern", "")
        return f" → {pat}" if pat else ""

    if tool_name in ("Task", "WebFetch", "WebSearch"):
        prompt = tool_input.get("prompt", tool_input.get("query", ""))
        return f" → {prompt[:100]}" if prompt else ""

    # Fallback: show first key/value pair
    for k, v in tool_input.items():
        return f" → {k}: {str(v)[:100]}"
    return ""


# ── Core converter ─────────────────────────────────────────────────────────────

def convert_jsonl(jsonl_path: Path, out_path: Path):
    """Read a single JSONL conversation file and write a clean .txt transcript."""

    with open(jsonl_path, encoding="utf-8") as f:
        raw_lines = f.readlines()

    messages = []
    for raw in raw_lines:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        mtype = msg.get("type", "")
        if mtype not in ("user", "assistant"):
            continue

        role       = msg.get("message", {}).get("role", mtype)
        content    = msg.get("message", {}).get("content", "")
        timestamp  = msg.get("timestamp", "")
        is_side    = msg.get("isSidechain", False)

        text = extract_text_from_content(content)
        if not text:
            continue

        messages.append({
            "role":      role,
            "text":      text,
            "timestamp": timestamp,
            "sidechain": is_side,
        })

    if not messages:
        print(f"  No messages found in {jsonl_path.name} — skipping.")
        return

    # Determine conversation date range
    ts_list = [m["timestamp"] for m in messages if m["timestamp"]]
    date_start = fmt_ts(min(ts_list)) if ts_list else "unknown"
    date_end   = fmt_ts(max(ts_list)) if ts_list else "unknown"

    session_id = jsonl_path.stem

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write(f"CONVERSATION TRANSCRIPT\n")
        out.write(f"Session ID : {session_id}\n")
        out.write(f"Start      : {date_start}\n")
        out.write(f"End        : {date_end}\n")
        out.write(f"Messages   : {len(messages)}\n")
        out.write("=" * 80 + "\n\n")

        for i, m in enumerate(messages, 1):
            role_label = "USER" if m["role"] == "user" else "CLAUDE"
            ts_label   = f"  [{fmt_ts(m['timestamp'])}]" if m["timestamp"] else ""
            side_label = "  [sidechain]" if m["sidechain"] else ""

            out.write(f"{'─' * 80}\n")
            out.write(f"[{i:04d}] {role_label}{ts_label}{side_label}\n")
            out.write(f"{'─' * 80}\n")
            out.write(m["text"])
            out.write("\n\n")

    size_kb = out_path.stat().st_size / 1024
    print(f"  Wrote {len(messages)} messages -> {out_path.name}  ({size_kb:.0f} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    jsonl_files = sorted(PROJECT_DIR.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {PROJECT_DIR}")
        return

    print(f"Found {len(jsonl_files)} conversation file(s).\n")

    for jsonl_path in jsonl_files:
        # Load to get date for a more descriptive filename
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
        except Exception:
            lines = []

        ts_values = [
            m.get("timestamp", "")
            for m in lines
            if m.get("type") in ("user", "assistant") and m.get("timestamp")
        ]
        if ts_values:
            date_str = datetime.fromisoformat(
                min(ts_values).replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
        else:
            date_str = "unknown-date"

        out_name = f"{date_str}_{jsonl_path.stem[:8]}.txt"
        out_path = OUTPUT_DIR / out_name

        print(f"Processing: {jsonl_path.name}")
        convert_jsonl(jsonl_path, out_path)

    print(f"\nDone. Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

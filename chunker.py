from __future__ import annotations

import html
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any

import fitz  # PyMuPDF


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
def load_config() -> dict[str, Any]:
    """读取 config.json。"""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"未找到配置文件：{CONFIG_PATH}\n"
            "请将 config.json 放在 chunker.py 同一目录下。"
        )
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)
CONFIG = load_config()
PROJECT_ROOT = Path(
    CONFIG["paths"]["project_root"]
).expanduser()
if not PROJECT_ROOT.is_absolute():
    PROJECT_ROOT = (SCRIPT_DIR / PROJECT_ROOT).resolve()
def resolve_path(path_text: str) -> Path:
    """绝对路径直接使用，相对路径相对于项目根目录解析。"""
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()
INPUT_DIR = resolve_path(CONFIG["paths"]["input_dir"])
OUTPUT_DIR = resolve_path(CONFIG["paths"]["output_dir"])
INTERMEDIATE_DIR = resolve_path(
    CONFIG["paths"]["intermediate_dir"]
)
OUTPUT_CONFIG = CONFIG["outputs"]
CHUNKS_JSONL = OUTPUT_DIR / OUTPUT_CONFIG["jsonl"]
CHUNKS_HTML = OUTPUT_DIR / OUTPUT_CONFIG["html"]
PAGE_TEXT_JSONL = (
    INTERMEDIATE_DIR / OUTPUT_CONFIG["page_text_jsonl"]
)
CHUNK_SIZE = int(
    CONFIG.get("chunking", {}).get("chunk_size", 300)
)
CHUNK_OVERLAP = int(
    CONFIG.get("chunking", {}).get("chunk_overlap", 60)
)
NORMALIZE_WHITESPACE = bool(
    CONFIG.get("chunking", {}).get(
        "normalize_whitespace", True
    )
)
PDF_PREFIX = CONFIG["ids"]["pdf_prefix"]
PDF_DIGITS = int(CONFIG["ids"]["pdf_digits"])
CHUNK_DIGITS = int(CONFIG["ids"]["chunk_digits"])
MIN_TEXT_WARNING = int(
    CONFIG["pdf"]["min_text_chars_per_page_warning"]
)
CONTINUE_ON_ERROR = bool(
    CONFIG["processing"]["continue_on_pdf_error"]
)

# ============================================================
# 2. 多级标点定义
# ============================================================
# 一级：句末标点（分句只用这一级）
PRIMARY_PUNCT = frozenset("。！？.!?")
# 二级：分句标点（单句超长时的降级）
SECONDARY_PUNCT = frozenset("；;：:")
# 三级：短停顿标点（再降级）
TERTIARY_PUNCT = frozenset("，,、")
# 优先级从高到低
PUNCT_LEVELS = (
    PRIMARY_PUNCT,
    SECONDARY_PUNCT,
    TERTIARY_PUNCT,
)
# ============================================================
# 3. 文本处理
# ============================================================
def normalize_text(text: str) -> str:
    """
    清理 PDF 提取文本：
      1. 去掉空字符
      2. 统一换行符
      3. 去掉所有 \\n 和 \\n\\n（PDF 换行不是语义边界）
      4. 压缩多余空格
    """
    text = text.replace("\x00", "")
    if NORMALIZE_WHITESPACE:
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = text.replace("\n", "")
        text = text.strip()
    return text
# ------------------------------------------------------------
# 3.1 按指定标点集合切分（保留标点）
# ------------------------------------------------------------
def _split_by_punct_set(
    text: str,
    offset: int,
    punct_set: frozenset,
) -> list[tuple[str, int, int]]:
    """按给定标点集合把 text 切成 [(片段, start, end)]，片段保留标点。"""
    pieces: list[tuple[str, int, int]] = []
    buf_start = 0
    for i, ch in enumerate(text):
        if ch in punct_set:
            piece = text[buf_start : i + 1]
            if piece.strip():
                pieces.append(
                    (piece, offset + buf_start, offset + i + 1)
                )
            buf_start = i + 1
    if buf_start < len(text):
        piece = text[buf_start:]
        if piece.strip():
            pieces.append(
                (piece, offset + buf_start, offset + len(text))
            )
    return pieces
# ------------------------------------------------------------
# 3.2 分句：只用第一级标点
# ------------------------------------------------------------
def split_into_sentences(
    text: str,
    offset: int = 0,
) -> list[tuple[str, int, int]]:
    """
    仅按第一级标点（。！？.!?）分句。

    返回 [(句文本, start, end), ...]，每句保留末尾标点。
    """
    return _split_by_punct_set(text, offset, PRIMARY_PUNCT)
# ------------------------------------------------------------
# 3.3 超长句的降级切割
# ------------------------------------------------------------
def _force_cut(
    text: str,
    offset: int,
    chunk_size: int,
) -> list[tuple[str, int, int]]:
    """硬切：按 chunk_size 切字符窗口。"""
    result: list[tuple[str, int, int]] = []
    for i in range(0, len(text), chunk_size):
        piece = text[i : i + chunk_size]
        if piece.strip():
            result.append(
                (piece, offset + i, offset + i + len(piece))
            )
    return result
def _split_long_sentence(
    text: str,
    offset: int,
    chunk_size: int,
    level: int = 1,
) -> list[tuple[str, int, int]]:
    """
    对**单句超长**的文本降级切割。

    策略：
      - 从 level=1（二级标点 ；;：:）开始尝试
      - 若该级切不动（只切出 1 片），降级到下一级
      - 三级标点也切不动 → 硬切

    返回 [(片段, start, end), ...]，每片长度尽量 ≤ chunk_size。
    """
    if len(text) <= chunk_size:
        if text.strip():
            return [(text, offset, offset + len(text))]
        return []

    if level >= len(PUNCT_LEVELS):
        return _force_cut(text, offset, chunk_size)

    punct_set = PUNCT_LEVELS[level]
    pieces = _split_by_punct_set(text, offset, punct_set)

    # 该级切不动 → 降级
    if len(pieces) <= 1:
        return _split_long_sentence(
            text, offset, chunk_size, level + 1
        )

    # 递归处理每一片（可能某些片还是超长）
    result: list[tuple[str, int, int]] = []
    for piece_text, s, e in pieces:
        result.extend(
            _split_long_sentence(
                piece_text, s, chunk_size, level + 1
            )
        )
    return result
# ------------------------------------------------------------
# 3.4 合并 + 处理超限
# ------------------------------------------------------------
def _merge_with_overlap(
    sentences: list[tuple[str, int, int]],
    chunk_size: int,
    overlap: int,
) -> list[list[tuple[str, int, int]]]:
    chunks: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    current_len = 0
    def flush(prev: list[tuple[str, int, int]]):
        """把 prev 作为一个 chunk 输出。"""
        if prev:
            chunks.append(prev)

    for sent in sentences:
        s_text, s_start, s_end = sent
        s_len = len(s_text)
        # ---------- 情况 ①：单句超长 ----------
        if s_len > chunk_size:
            # 先 flush 已有的块
            if current:
                flush(current)
                current = []
                current_len = 0
            # 对这句降级切割（从二级标点开始）
            sub_pieces = _split_long_sentence(
                s_text, s_start, chunk_size, level=1
            )
            # 除最后一片外，都直接成为独立 chunk
            for p in sub_pieces[:-1]:
                chunks.append([p])
            # 最后一片留在缓冲区，等待后续合并
            if sub_pieces:
                current = [sub_pieces[-1]]
                current_len = len(sub_pieces[-1][0])
            continue
        # ---------- 情况 ②：累加超限 or 能塞下 ----------
        if not current:
            current = [sent]
            current_len = s_len
            continue
        if current_len + s_len <= chunk_size:
            # 能塞 → 合并
            current.append(sent)
            current_len += s_len
            continue
        # 塞不下 → 当前句留给下一块
        prev = current
        chunks.append(prev)
        # 计算尾部 overlap 长度
        prev_text = "".join(t for t, _, _ in prev)
        prev_end = prev[-1][2]
        # 保证新块不超限：最多能带多少字
        max_tail = max(0, chunk_size - s_len)
        tail_len = min(overlap, len(prev_text), max_tail)
        if tail_len > 0:
            tail_text = prev_text[-tail_len:]
            tail_start = prev_end - tail_len
            new_text = tail_text + s_text
            current = [(new_text, tail_start, s_end)]
            current_len = len(new_text)
        else:
            current = [sent]
            current_len = s_len
    # 结尾 flush
    if current:
        chunks.append(current)
    return chunks

def split_text_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    if not text:
        return []

    # 短文本直接返回
    if len(text) <= chunk_size:
        stripped = text.strip()
        if not stripped:
            return []
        leading = len(text) - len(text.lstrip())
        trailing = len(text) - len(text.rstrip())
        return [
            (stripped, leading, len(text) - trailing)
        ]

    # ① 分句（只用第一级标点）
    sentences = split_into_sentences(text, 0)

    # ② 合并 + 处理超限
    packed = _merge_with_overlap(
        sentences, chunk_size, overlap
    )

    # ③ 打包成 (文本, start, end)
    result: list[tuple[str, int, int]] = []
    for group in packed:
        chunk_text = "".join(t for t, _, _ in group)
        chunk_start = group[0][1]
        chunk_end = group[-1][2]
        if chunk_text.strip():
            result.append(
                (chunk_text, chunk_start, chunk_end)
            )
    return result

def build_chunks_for_pdf(
    pages: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """对整个 PDF 所有页面整体切分（允许跨页）。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap 必须大于等于 0，且小于 chunk_size。"
        )
    # -------- 拼接全文 + 记录每页位置 --------
    parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []
    cursor = 0

    for page_record in pages:
        page_num = int(page_record["page"])
        text = normalize_text(page_record.get("page_text", ""))

        if not text:
            continue

        page_start = cursor
        parts.append(text)
        cursor += len(text)
        page_spans.append((page_num, page_start, cursor))

        parts.append(" ")
        cursor += 1

    full_text = "".join(parts)

    if not full_text.strip():
        return []

    # -------- 切分 --------
    pieces = split_text_into_chunks(
        full_text, chunk_size, overlap
    )
    # -------- 反查页码 --------
    chunks: list[dict[str, Any]] = []

    for piece_text, p_start, p_end in pieces:
        page_list: list[int] = []

        for page_num, s_start, s_end in page_spans:
            if s_start < p_end and s_end > p_start:
                if page_num not in page_list:
                    page_list.append(page_num)

        if not page_list:
            continue

        chunks.append(
            {
                "text": piece_text,
                "pages": page_list,
                "page_start": page_list[0],
                "page_end": page_list[-1],
            }
        )

    return chunks

# ============================================================
# 4. JSONL 写入工具
# ============================================================
def write_jsonl_record(file, record: dict[str, Any]) -> None:
    file.write(
        json.dumps(record, ensure_ascii=False) + "\n"
    )

# ============================================================
# 5. PDF 扫描与逐页提取
# ============================================================
def discover_pdfs() -> list[Path]:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"PDF 输入目录不存在：{INPUT_DIR}\n"
            "请检查 config.json 中的 paths.input_dir。"
        )
    pdf_files = [
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    pdf_files.sort(
        key=lambda path: path.relative_to(INPUT_DIR)
        .as_posix()
        .lower()
    )
    return pdf_files
def make_pdf_id(index: int) -> str:
    return f"{PDF_PREFIX}_{index:0{PDF_DIGITS}d}"
def make_chunk_id(pdf_id: str, chunk_index: int) -> str:
    return f"{pdf_id}_CHUNK_{chunk_index:0{CHUNK_DIGITS}d}"
def extract_pdf_pages(
    pdf_path: Path,
    pdf_id: str,
    page_output_file,
) -> tuple[list[dict[str, Any]], int]:
    pages = []
    relative_path = pdf_path.relative_to(INPUT_DIR).as_posix()
    with fitz.open(pdf_path) as document:
        total_pages = len(document)
        for page_index in range(total_pages):
            page = document.load_page(page_index)

            page_text = page.get_text(
                "text",
                sort=bool(CONFIG["pdf"]["sort_text_blocks"]),
            )
            page_text = normalize_text(page_text)
            page_number = page_index + 1
            page_record = {
                "pdf_id": pdf_id,
                "pdf_name": pdf_path.name,
                "source_file": relative_path,
                "page": page_number,
                "page_text": page_text,
                "char_count": len(page_text),
            }
            pages.append(page_record)
            write_jsonl_record(page_output_file, page_record)
    return pages, total_pages
# ============================================================
# 6. 生成 chunks.jsonl
# ============================================================
def build_chunks(
    pdf_files: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_chunks: list[dict[str, Any]] = []
    pdf_manifest: list[dict[str, Any]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    with (
        CHUNKS_JSONL.open(
            "w", encoding="utf-8", newline="\n"
        ) as chunks_file,
        PAGE_TEXT_JSONL.open(
            "w", encoding="utf-8", newline="\n"
        ) as page_file,
    ):
        for pdf_index, pdf_path in enumerate(pdf_files, start=1):
            pdf_id = make_pdf_id(pdf_index)
            relative_path = pdf_path.relative_to(
                INPUT_DIR
            ).as_posix()
            try:
                pages, total_pages = extract_pdf_pages(
                    pdf_path,
                    pdf_id,
                    page_file,
                )
                pdf_chunks = build_chunks_for_pdf(
                    pages,
                    CHUNK_SIZE,
                    CHUNK_OVERLAP,
                )
                for chunk_index, chunk in enumerate(
                    pdf_chunks, start=1
                ):
                    chunk_record = {
                        "pdf_id": pdf_id,
                        "chunk_id": make_chunk_id(
                            pdf_id, chunk_index
                        ),
                        "pdf_name": pdf_path.name,
                        "source_file": relative_path,
                        "pages": chunk["pages"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "text": chunk["text"],
                        "char_count": len(chunk["text"]),
                    }
                    all_chunks.append(chunk_record)
                    write_jsonl_record(
                        chunks_file, chunk_record
                    )
                pdf_manifest.append(
                    {
                        "pdf_id": pdf_id,
                        "pdf_name": pdf_path.name,
                        "source_file": relative_path,
                        "page_count": total_pages,
                        "chunk_count": len(pdf_chunks),
                        "status": "success",
                    }
                )
            except Exception as exc:
                pdf_manifest.append(
                    {
                        "pdf_id": pdf_id,
                        "pdf_name": pdf_path.name,
                        "source_file": relative_path,
                        "page_count": 0,
                        "chunk_count": 0,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                if not CONTINUE_ON_ERROR:
                    raise

    manifest_path = OUTPUT_DIR / "pdf_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            pdf_manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return all_chunks, pdf_manifest

def format_page_display(pages: list[int]) -> str:
    if not pages:
        return "无页码"

    if len(pages) == 1:
        return f"第 {pages[0]} 页"

    if pages[-1] - pages[0] + 1 == len(pages):
        return f"第 {pages[0]}-{pages[-1]} 页"

    return "第 " + "、".join(str(p) for p in pages) + " 页"


def generate_html(
    chunks: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> None:
    title = html.escape(
        CONFIG.get("preview", {}).get(
            "title",
            "PDF Chunk Preview",
        )
    )

    manifest_by_id = {
        item["pdf_id"]: item
        for item in manifest
    }

    chunks_by_pdf: dict[str, list[dict[str, Any]]] = {}

    for chunk in chunks:
        chunks_by_pdf.setdefault(
            chunk["pdf_id"], []
        ).append(chunk)

    pdf_options = []

    for item in manifest:
        option_label = (
            f'{item["pdf_id"]} | {item["pdf_name"]}'
        )
        pdf_options.append(
            f'<option value="{html.escape(item["pdf_id"], quote=True)}">'
            f'{html.escape(option_label)}</option>'
        )

    pdf_sections = []

    for pdf_id, pdf_chunks in chunks_by_pdf.items():
        pdf_info = manifest_by_id[pdf_id]
        pdf_name = pdf_info["pdf_name"]

        cards = []

        for chunk in pdf_chunks:
            chunk_id = html.escape(
                chunk["chunk_id"], quote=True,
            )
            chunk_pages = list(chunk.get("pages", []))
            pages_attr = ",".join(str(p) for p in chunk_pages)
            page_display = html.escape(
                format_page_display(chunk_pages)
            )
            char_count = chunk["char_count"]
            source_file = html.escape(
                chunk["source_file"]
            )
            chunk_text = html.escape(chunk["text"])

            searchable = html.escape(
                f'{chunk["pdf_name"]} {chunk["pdf_id"]} '
                f'{chunk["chunk_id"]} {pages_attr} '
                f'{chunk["text"]}',
                quote=True,
            )

            cards.append(
                f"""
                <article class="chunk-card"
                    data-pdf="{html.escape(pdf_id, quote=True)}"
                    data-pages="{pages_attr}"
                    data-search="{searchable.lower()}">

                    <div class="chunk-header">
                        <div class="chunk-identifiers">
                            <span class="chunk-id">{chunk_id}</span>
                            <span class="tag">{page_display}</span>
                            <span class="tag">{char_count} 字符</span>
                        </div>

                        <div class="chunk-actions">
                            <button class="small-button toggle-button"
                                type="button"
                                onclick="toggleChunk(this)">
                                展开 / 收起
                            </button>

                            <button class="small-button copy-button"
                                type="button"
                                onclick="copyChunk(this)">
                                复制原文
                            </button>
                        </div>
                    </div>

                    <div class="source-line">
                        来源：{source_file}
                    </div>

                    <pre class="chunk-text">{chunk_text}</pre>
                </article>
                """
            )

        section = f"""
        <section class="pdf-section"
            data-pdf-section="{html.escape(pdf_id, quote=True)}">

            <div class="pdf-heading">
                <div>
                    <div class="pdf-id">{html.escape(pdf_id)}</div>
                    <h2>{html.escape(pdf_name)}</h2>
                    <div class="pdf-meta">
                        {pdf_info["page_count"]} 页
                        · {pdf_info["chunk_count"]} 个 chunk
                    </div>
                    <div class="source-line">
                        {html.escape(pdf_info["source_file"])}
                    </div>
                </div>
                <button class="outline-button"
                    type="button"
                    onclick="togglePdf(this)">
                    展开 / 收起本篇
                </button>
            </div>

            <div class="chunk-list">
                {''.join(cards)}
            </div>
        </section>
        """

        pdf_sections.append(section)

    failed_count = sum(
        1 for item in manifest
        if item.get("status") != "success"
    )

    generated_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    document_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>

<style>
:root {{
    --bg: #f4f6fb;
    --surface: #ffffff;
    --text: #1f2937;
    --muted: #667085;
    --border: #e5e9f2;
    --accent: #4f46e5;
    --accent-light: #eef2ff;
    --success: #15803d;
    --shadow: 0 6px 24px rgba(16, 24, 40, 0.06);
}}

* {{ box-sizing: border-box; }}

body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    line-height: 1.6;
}}

.container {{
    max-width: 1240px;
    margin: 0 auto;
    padding: 32px 24px 60px;
}}

.hero {{
    padding: 30px 32px;
    background: linear-gradient(135deg, #252d63, #4f46e5);
    color: white;
    border-radius: 20px;
    box-shadow: var(--shadow);
    margin-bottom: 24px;
}}

.hero h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: -0.5px; }}
.hero p {{ margin: 0; opacity: 0.86; }}

.stats {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 24px;
}}

.stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
}}

.stat-label {{ color: var(--muted); font-size: 13px; }}
.stat-value {{ font-size: 27px; font-weight: 700; margin-top: 4px; }}

.toolbar {{
    position: sticky;
    top: 0;
    z-index: 10;
    display: grid;
    grid-template-columns: minmax(220px, 1.8fr)
                           minmax(180px, 1fr)
                           minmax(120px, 0.6fr)
                           auto;
    gap: 12px;
    align-items: center;
    padding: 16px;
    margin-bottom: 24px;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
}}

input, select {{
    width: 100%;
    min-width: 0;
    padding: 11px 12px;
    border: 1px solid #d0d5dd;
    border-radius: 9px;
    background: white;
    color: var(--text);
    font-size: 14px;
    outline: none;
}}

input:focus, select:focus {{
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
}}

button {{ cursor: pointer; font-family: inherit; }}

.outline-button, .small-button {{
    border: 1px solid #d8dcf0;
    border-radius: 8px;
    padding: 8px 12px;
    background: white;
    color: #3730a3;
    font-size: 13px;
    white-space: nowrap;
}}

.outline-button:hover, .small-button:hover {{
    background: var(--accent-light);
}}

.pdf-section {{
    margin-bottom: 22px;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--surface);
    box-shadow: var(--shadow);
    overflow: hidden;
}}

.pdf-heading {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 22px 24px;
    background: #fbfcff;
    border-bottom: 1px solid var(--border);
}}

.pdf-id {{
    color: var(--accent);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.1px;
    text-transform: uppercase;
}}

.pdf-heading h2 {{
    margin: 4px 0 5px;
    font-size: 20px;
    overflow-wrap: anywhere;
}}

.pdf-meta, .source-line {{
    color: var(--muted);
    font-size: 12px;
    overflow-wrap: anywhere;
}}

.chunk-list {{ padding: 16px; }}

.chunk-card {{
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 12px;
    overflow: hidden;
}}

.chunk-card:last-child {{ margin-bottom: 0; }}

.chunk-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    padding: 13px 16px;
    background: #fff;
}}

.chunk-identifiers, .chunk-actions {{
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}}

.chunk-id {{
    color: #3730a3;
    font-weight: 700;
    font-size: 13px;
    overflow-wrap: anywhere;
}}

.tag {{
    display: inline-block;
    border-radius: 6px;
    padding: 3px 8px;
    background: #f2f4f7;
    color: #475467;
    font-size: 12px;
}}

.source-line {{ padding: 0 16px 10px; }}

.chunk-text {{
    margin: 0;
    padding: 16px;
    border-top: 1px solid #f0f1f5;
    background: #fcfcfe;
    color: #344054;
    font-family: "Cascadia Code", Consolas, "Microsoft YaHei", monospace;
    font-size: 13px;
    line-height: 1.85;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}}

.chunk-text.collapsed {{ display: none; }}

.empty-state {{
    padding: 44px 20px;
    text-align: center;
    color: var(--muted);
    background: white;
    border: 1px dashed #cbd5e1;
    border-radius: 14px;
}}

.footer {{
    color: var(--muted);
    font-size: 12px;
    text-align: center;
    margin-top: 28px;
}}

.toast {{
    position: fixed;
    right: 24px;
    bottom: 24px;
    padding: 12px 18px;
    background: #1f2937;
    color: white;
    border-radius: 10px;
    box-shadow: var(--shadow);
    opacity: 0;
    transform: translateY(8px);
    transition: all 0.2s ease;
    pointer-events: none;
}}

.toast.show {{ opacity: 1; transform: translateY(0); }}
.hidden {{ display: none !important; }}

@media (max-width: 850px) {{
    .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .toolbar {{ position: static; grid-template-columns: 1fr; }}
    .container {{ padding: 16px 12px 36px; }}
    .hero {{ padding: 24px 20px; }}
    .hero h1 {{ font-size: 24px; }}
    .pdf-heading {{ align-items: flex-start; flex-direction: column; }}
}}
</style>
</head>

<body>
<div class="container">

    <header class="hero">
        <h1>{title}</h1>
        <p>PDF 文献切分检查面板 · 按 PDF、页码和 chunk 追溯原文</p>
        <p style="margin-top:10px;font-size:12px;">
            生成时间：{generated_time}
        </p>
    </header>

    <section class="stats">
        <div class="stat">
            <div class="stat-label">PDF 总数</div>
            <div class="stat-value">{len(manifest)}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Chunk 总数</div>
            <div class="stat-value">{len(chunks)}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Chunk 字符上限</div>
            <div class="stat-value">{CHUNK_SIZE}</div>
        </div>
        <div class="stat">
            <div class="stat-label">处理失败的 PDF</div>
            <div class="stat-value">{failed_count}</div>
        </div>
    </section>

    <section class="toolbar">
        <input id="searchInput"
            type="search"
            placeholder="搜索 PDF 名称、chunk ID 或正文..."
            oninput="filterChunks()">

        <select id="pdfFilter" onchange="filterChunks()">
            <option value="">全部 PDF</option>
            {''.join(pdf_options)}
        </select>

        <input id="pageFilter"
            type="number"
            min="1"
            placeholder="页码筛选"
            oninput="filterChunks()">

        <button class="outline-button"
            type="button"
            onclick="resetFilters()">
            重置筛选
        </button>
    </section>

    <div id="resultSummary"
        style="color:#667085;font-size:13px;margin:0 2px 14px;">
        当前显示 {len(chunks)} 个 chunk
    </div>

    <main id="pdfContainer">
        {''.join(pdf_sections)}
    </main>

    <div id="emptyState" class="empty-state hidden">
        没有找到匹配的 chunk，请调整搜索词或筛选条件。
    </div>

    <footer class="footer">
        PDF Chunk Preview · 输出用于 RAG 数据检查与后续向量数据库入库
    </footer>
</div>

<div id="toast" class="toast">已复制</div>

<script>
function parseCardPages(card) {{
    const raw = card.dataset.pages || "";
    return raw
        .split(",")
        .map(item => Number(item.trim()))
        .filter(value => Number.isFinite(value) && value > 0);
}}

function filterChunks() {{
    const query = document.getElementById("searchInput")
        .value.trim().toLowerCase();

    const selectedPdf = document.getElementById("pdfFilter").value;
    const pageValue = document.getElementById("pageFilter").value;
    const page = pageValue ? Number(pageValue) : null;

    let visibleCount = 0;

    document.querySelectorAll(".chunk-card").forEach(card => {{
        const searchText = card.dataset.search || "";
        const pdfId = card.dataset.pdf || "";
        const cardPages = parseCardPages(card);

        const matchesQuery = !query || searchText.includes(query);
        const matchesPdf = !selectedPdf || pdfId === selectedPdf;
        const matchesPage = page === null || cardPages.includes(page);

        const visible = matchesQuery && matchesPdf && matchesPage;

        card.classList.toggle("hidden", !visible);

        if (visible) visibleCount++;
    }});

    document.querySelectorAll(".pdf-section").forEach(section => {{
        const visibleCards = section.querySelectorAll(
            ".chunk-card:not(.hidden)"
        ).length;

        section.classList.toggle("hidden", visibleCards === 0);
    }});

    document.getElementById("resultSummary").textContent =
        "当前显示 " + visibleCount + " 个 chunk";

    document.getElementById("emptyState").classList.toggle(
        "hidden",
        visibleCount !== 0
    );
}}

function resetFilters() {{
    document.getElementById("searchInput").value = "";
    document.getElementById("pdfFilter").value = "";
    document.getElementById("pageFilter").value = "";
    filterChunks();
}}

function toggleChunk(button) {{
    const card = button.closest(".chunk-card");
    const text = card.querySelector(".chunk-text");
    text.classList.toggle("collapsed");
}}

function togglePdf(button) {{
    const section = button.closest(".pdf-section");
    const cards = section.querySelectorAll(".chunk-text");

    const hasVisibleText = Array.from(cards).some(
        text => !text.classList.contains("collapsed")
    );

    cards.forEach(text => {{
        text.classList.toggle("collapsed", hasVisibleText);
    }});
}}

async function copyChunk(button) {{
    const card = button.closest(".chunk-card");
    const text = card.querySelector(".chunk-text").textContent;

    try {{
        await navigator.clipboard.writeText(text);
        showToast("Chunk 原文已复制");
    }} catch (error) {{
        const textarea = document.createElement("textarea");
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();

        try {{
            document.execCommand("copy");
            showToast("Chunk 原文已复制");
        }} catch (fallbackError) {{
            showToast("复制失败，请手动选择文本");
        }}

        textarea.remove();
    }}
}}

let toastTimer;

function showToast(message) {{
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.classList.add("show");

    clearTimeout(toastTimer);

    toastTimer = setTimeout(() => {{
        toast.classList.remove("show");
    }}, 1800);
}}

document.querySelectorAll(".chunk-text").forEach(text => {{
    text.classList.add("collapsed");
}});

filterChunks();
</script>
</body>
</html>
"""

    with CHUNKS_HTML.open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        file.write(document_html)


def main() -> None:
    pdf_files = discover_pdfs()

    if not pdf_files:
        return

    chunks, manifest = build_chunks(pdf_files)

    generate_html(chunks, manifest)

    print("文件已经全部处理")


if __name__ == "__main__":
    main()
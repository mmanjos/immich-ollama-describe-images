#!/usr/bin/env python3
# https://github.com/mmanjos/immich-ollama-describe-images

__author__ = "github.com/mmanjos"
__copyright__ = "Copyright 2026, Matthew Manjos"
__license__ = "GPL-3"
__version__ = "0.0.1"
__maintainer__ = "Matthew Manjos"
__email__ = "matt@manjos.com"

"""Describes Immich photos with a local Ollama vision model and write the result
back into each asset's description metadata field."""

import argparse
import contextlib
import json
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import ollama
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# DEFAULT_SERVER = "https://set.to.your.immich.url"
# DEFAULT_MODEL = "llama3.2-vision"
# DEFAULT_MODEL = "qwen3.5:9b"
# DEFAULT_MODEL = "gemma3:12b"
DEFAULT_MODEL = "gemma3:12b-it-qat"
# DEFAULT_MODEL = "gemma3:27b"
# DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_PROMPT = "CRITICAL: Do not include a preamble or introductory sentence. Describe this image in 5 sentences or fewer. Start with the most important visual element. Use descriptive adjectives and avoid filler phrases like 'In this image' or 'I can see.' Avoid commenting on the mood or emotion of the image. Be direct and high-density."
OLLAMA_OPTIONS = {"num_predict": 1000, "temperature": 0.1}
PAGE_SIZE = 250
HTTP_TIMEOUT = 120
JOURNAL_DIR = Path.home() / ".local" / "share" / "immich-ollama-describe-images"
DISPLAY_TAIL = 15
PREFETCH = 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    server_default = globals().get("DEFAULT_SERVER")
    model_default = globals().get("DEFAULT_MODEL")
    prompt_default = globals().get("DEFAULT_PROMPT")
    p.add_argument("--server", default=server_default, required=not server_default, help="Immich server URL")
    p.add_argument("--model", default=model_default, required=not model_default, help="Ollama model name")
    p.add_argument(
        "--prompt", default=prompt_default, required=not prompt_default, help="Prompt sent to the model"
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N new assets this run")
    p.add_argument(
        "--benchmark",
        nargs="?",
        const=True,
        default=None,
        metavar="ASSET_ID",
        help="Run the prompt against every locally-available Ollama model. Without an argument, "
        "benchmarks the first 8 Immich images and writes per-image results to "
        "./benchmark-<filename>.txt. With an Immich asset ID, benchmarks just that one image and "
        "prints results to the terminal. Does not modify Immich.",
    )
    return p.parse_args()


def get_me(client):
    r = client.get("/api/users/me")
    r.raise_for_status()
    return r.json()


def list_all_assets(client):
    """Walk every page of /api/search/metadata and return (id, filename) pairs.

    The `total` field on each page reflects the page size, not the library size,
    so the only reliable count is to enumerate. We keep just id+filename to bound
    memory for large libraries.
    """
    out = []
    page = 1
    while True:
        r = client.post(
            "/api/search/metadata",
            json={"page": page, "size": PAGE_SIZE, "type": "IMAGE"},
        )
        r.raise_for_status()
        block = r.json().get("assets", {})
        for item in block.get("items", []):
            aid = item.get("id")
            if not aid:
                continue
            out.append((aid, item.get("originalFileName") or aid))
        nxt = block.get("nextPage")
        if not nxt:
            return out
        page = int(nxt)


# TIFF and the common camera RAW formats (DNG/NEF/CR2/CR3/ARW/ORF/RW2/RAF/PEF)
# are all TIFF-based containers; ollama's image decoder rejects them with
# "tiff: unsupported feature: color model" or similar.
PREVIEW_FALLBACK_EXTS = (
    ".tif",
    ".tiff",
    ".dng",
    ".nef",
    ".cr2",
    ".cr3",
    ".arw",
    ".orf",
    ".rw2",
    ".raf",
    ".pef",
)


def needs_preview_fallback(fname):
    return fname.lower().endswith(PREVIEW_FALLBACK_EXTS)


def download_image(client, asset_id, fname):
    # For formats ollama can't decode, pull the Immich-rendered preview JPG
    # (`size=preview` is the largest thumbnail) instead of the original bytes.
    if needs_preview_fallback(fname):
        r = client.get(f"/api/assets/{asset_id}/thumbnail", params={"size": "preview"})
    else:
        r = client.get(f"/api/assets/{asset_id}/original")
    r.raise_for_status()
    return r.content


def set_description(client, asset_id, description):
    r = client.put(f"/api/assets/{asset_id}", json={"description": description})
    r.raise_for_status()


def load_journal(path):
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_journal(path, journal):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(journal, f, indent=2, sort_keys=True)
    tmp.replace(path)


def safe_filename(s):
    return re.sub(r"[^A-Za-z0-9._@+-]+", "_", s).strip("_") or "account"


def journal_matches(entry, model, prompt, temperature):
    return (
        bool(entry)
        and entry.get("model") == model
        and entry.get("prompt") == prompt
        and entry.get("temperature") == temperature
    )


def format_duration(seconds):
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def list_ollama_models():
    """Return the names of locally-installed Ollama models.

    The python ollama client has shipped both dict-style and object-style
    responses across versions, so handle both rather than pinning a version.
    """
    response = ollama.list()
    raw = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
    names = []
    for m in raw:
        if isinstance(m, dict):
            name = m.get("model") or m.get("name")
        else:
            name = getattr(m, "model", None) or getattr(m, "name", None)
        if name:
            names.append(name)
    return names


def fetch_first_assets(client, count):
    r = client.post(
        "/api/search/metadata",
        json={"page": 1, "size": count, "type": "IMAGE"},
    )
    r.raise_for_status()
    items = r.json().get("assets", {}).get("items", [])[:count]
    return [(it["id"], it.get("originalFileName") or it["id"]) for it in items if it.get("id")]


def get_asset(client, asset_id):
    r = client.get(f"/api/assets/{asset_id}")
    r.raise_for_status()
    item = r.json()
    return (item["id"], item.get("originalFileName") or item["id"])


def run_benchmark(args, client, console, asset_id=None):
    """Run benchmark mode.

    asset_id=None: benchmark the first 8 images, write per-image .txt files.
    asset_id=<str>: benchmark just that asset, print results to the terminal.
    """
    console.print("[dim]listing locally-installed ollama models...[/dim]")
    try:
        models = list_ollama_models()
    except Exception as e:
        console.print(f"[red]error listing ollama models: {e}[/red]")
        sys.exit(2)
    if not models:
        console.print("[red]no ollama models available[/red]")
        sys.exit(2)
    console.print(f"[bold]Models ({len(models)}):[/bold]")
    for m in models:
        console.print(f"  - {m}")

    write_files = asset_id is None

    if write_files:
        console.print("\n[dim]fetching first 8 images from immich...[/dim]")
        try:
            images = fetch_first_assets(client, 8)
        except httpx.HTTPError as e:
            console.print(f"[red]error fetching images: {e}[/red]")
            sys.exit(2)
        if not images:
            console.print("[red]no images found in library[/red]")
            sys.exit(2)
    else:
        console.print(f"\n[dim]fetching asset {asset_id} from immich...[/dim]")
        try:
            images = [get_asset(client, asset_id)]
        except httpx.HTTPError as e:
            console.print(f"[red]error fetching asset {asset_id}: {e}[/red]")
            sys.exit(2)
    console.print(f"[bold]Images:[/bold] {len(images)}\n")

    out_dir = Path.cwd()
    total_calls = len(images) * len(models)
    call_idx = 0

    for aid, fname in images:
        console.print(f"[bold cyan]Image:[/bold cyan] {fname}")
        try:
            img_bytes = download_image(client, aid, fname)
        except httpx.HTTPError as e:
            console.print(f"  [red]download failed: {e}[/red]")
            continue

        out_path = out_dir / f"benchmark-{safe_filename(fname)}.txt" if write_files else None
        ctx = open(out_path, "w") if write_files else contextlib.nullcontext()
        with ctx as f:
            if f:
                f.write(f"# benchmark for {fname}\n")
                f.write(f"# asset id: {aid}\n")
                f.write(f"# prompt: {args.prompt}\n")
                f.write(f"# options: {json.dumps(OLLAMA_OPTIONS)}\n\n")

            for model in models:
                call_idx += 1
                console.print(f"  [{call_idx}/{total_calls}] [bold]{model}[/bold] ...", end="")
                start = time.time()
                try:
                    response = ollama.generate(
                        model=model,
                        prompt=args.prompt,
                        images=[img_bytes],
                        options=OLLAMA_OPTIONS,
                    )
                    description = response["response"].strip()
                    if not description:
                        raise ValueError(
                            f"empty response from model (done_reason={response.get('done_reason')!r})"
                        )
                    duration = time.time() - start
                    console.print(f" [green]ok[/green] ({duration:.1f}s)")
                    if f:
                        f.write(f"=== {model} ({duration:.1f}s) ===\n")
                        f.write(description + "\n\n")
                    else:
                        console.print(description)
                        console.print()
                except Exception as e:
                    duration = time.time() - start
                    console.print(f" [red]error[/red] ({duration:.1f}s): {e}")
                    if f:
                        f.write(f"=== {model} (ERROR after {duration:.1f}s) ===\n")
                        f.write(f"{e}\n\n")
                if f:
                    f.flush()
        if write_files:
            console.print(f"  [dim]wrote {out_path}[/dim]\n")

    console.print("[bold green]Benchmark complete.[/bold green]")


def render(state):
    elapsed = max(time.time() - state["start"], 1e-6)
    run_count = state["run_count"]
    rate = run_count / elapsed if run_count else 0.0
    total = state["total"]
    done = state["done_total"]
    remaining = max(total - done, 0)
    eta = format_duration(remaining / rate) if rate > 0 else "—"
    pct = (done / total * 100.0) if total else 0.0

    banner_text = Text.assemble(
        ("Library: ", "bold"),
        (f"{done}/{total} ", "cyan"),
        (f"({pct:.1f}%)", "dim"),
        "    ",
        ("rate: ", "bold"),
        (f"{rate * 60:.1f}/min", "magenta"),
        "    ",
        ("ETA: ", "bold"),
        (eta, "yellow"),
        "    ",
        ("this run: ", "bold"),
        (f"{run_count} ok / {len(state['errors'])} err", "white"),
    )
    banner = Panel(banner_text, title="Immich AI Describe", border_style="blue")

    cols = Table.grid(expand=True, padding=(0, 1))
    cols.add_column(ratio=1)
    cols.add_column(ratio=1)

    def panel_for(title, items, style):
        if not items:
            body = Text("(none yet)", style="dim")
        else:
            body = Text("\n".join(items[-DISPLAY_TAIL:]))
        return Panel(body, title=f"{title} ({len(items)})", border_style=style)

    cols.add_row(
        panel_for("Processed this run", state["success"], "green"),
        panel_for("Errors this run", state["errors"], "red"),
    )

    return Group(banner, cols)


def main():
    args = parse_args()
    api_key = os.environ.get("IMMICH_API_KEY")
    if not api_key:
        print("error: IMMICH_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    base = args.server.rstrip("/")
    headers = {"x-api-key": api_key, "Accept": "application/json"}
    console = Console()

    with httpx.Client(base_url=base, headers=headers, timeout=HTTP_TIMEOUT) as client:
        try:
            me = get_me(client)
        except httpx.HTTPError as e:
            console.print(f"[red]error contacting Immich at {base}: {e}[/red]")
            sys.exit(2)

        account = me.get("email") or me.get("name") or me.get("id") or "unknown"

        if args.benchmark is not None:
            asset_id = args.benchmark if isinstance(args.benchmark, str) else None
            console.print(f"[bold]Account:[/bold] {account}")
            console.print(f"[bold]Server:[/bold]  {base}")
            console.print(f"[bold]Prompt:[/bold]  {args.prompt}\n")
            run_benchmark(args, client, console, asset_id=asset_id)
            return

        journal_path = JOURNAL_DIR / f"{safe_filename(account)}.json"
        journal = load_journal(journal_path)

        console.print(f"[bold]Account:[/bold] {account}")
        console.print(f"[bold]Server:[/bold]  {base}")
        console.print(f"[bold]Model:[/bold]   {args.model}")
        console.print(f"[bold]Prompt:[/bold]  {args.prompt}")
        console.print(f"[bold]Journal:[/bold] {journal_path} ({len(journal)} entries)", highlight=False)
        console.print("[dim]enumerating library...[/dim]")

        try:
            all_assets = list_all_assets(client)
        except httpx.HTTPError as e:
            console.print(f"[red]error listing assets: {e}[/red]")
            sys.exit(2)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted during library enumeration.[/yellow]")
            sys.exit(130)

        total = len(all_assets)
        temperature = OLLAMA_OPTIONS.get("temperature")
        pending_assets = [
            (aid, fname)
            for aid, fname in all_assets
            if not journal_matches(journal.get(aid), args.model, args.prompt, temperature)
        ]
        already_done = total - len(pending_assets)
        console.print(
            f"[bold]Library:[/bold] {total} images " f"({already_done} already match current model+prompt)\n"
        )

        state = {
            "success": [],
            "errors": [],
            "total": total,
            "done_total": already_done,
            "run_count": 0,
            "start": time.time(),
        }
        state_lock = threading.Lock()
        journal_lock = threading.Lock()

        def do_upload(aid, fname, description):
            try:
                set_description(client, aid, description)
            except httpx.ReadTimeout:
                with state_lock:
                    state["errors"].append(f"{fname}: timeout")
                return
            except httpx.HTTPStatusError as e:
                with state_lock:
                    state["errors"].append(f"{fname}: http {e.response.status_code}")
                return
            except Exception as e:
                with state_lock:
                    state["errors"].append(f"{fname}: {e}")
                return
            with journal_lock:
                journal[aid] = {
                    "filename": fname,
                    "model": args.model,
                    "prompt": args.prompt,
                    "temperature": temperature,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }
                save_journal(journal_path, journal)
            with state_lock:
                state["success"].append(fname)
                state["done_total"] += 1
                state["run_count"] += 1

        interrupted = False
        with Live(render(state), console=console, refresh_per_second=4, transient=False) as live:
            dl_exec = ThreadPoolExecutor(max_workers=PREFETCH, thread_name_prefix="dl")
            ul_exec = ThreadPoolExecutor(max_workers=PREFETCH, thread_name_prefix="ul")
            inflight = deque()
            try:
                pending_iter = iter(pending_assets)

                def submit_next():
                    try:
                        nxt = next(pending_iter)
                    except StopIteration:
                        return False
                    aid, fname = nxt
                    inflight.append((aid, fname, dl_exec.submit(download_image, client, aid, fname)))
                    return True

                for _ in range(PREFETCH):
                    if not submit_next():
                        break

                dispatched = 0
                while inflight:
                    if args.limit is not None and dispatched >= args.limit:
                        break

                    aid, fname, fut = inflight.popleft()
                    submit_next()  # keep the prefetch pipe full

                    try:
                        img_bytes = fut.result()
                    except httpx.ReadTimeout:
                        with state_lock:
                            state["errors"].append(f"{fname}: timeout")
                        live.update(render(state))
                        continue
                    except httpx.HTTPStatusError as e:
                        with state_lock:
                            state["errors"].append(f"{fname}: http {e.response.status_code}")
                        live.update(render(state))
                        continue
                    except Exception as e:
                        with state_lock:
                            state["errors"].append(f"{fname}: {e}")
                        live.update(render(state))
                        continue

                    try:
                        response = ollama.generate(
                            model=args.model,
                            prompt=args.prompt,
                            images=[img_bytes],
                            options=OLLAMA_OPTIONS,
                        )
                        description = response["response"].strip()
                        if not description:
                            raise ValueError(
                                f"empty description from model (done_reason={response.get('done_reason')!r})"
                            )
                    except ollama.ResponseError as e:
                        with state_lock:
                            state["errors"].append(f"{fname}: ollama {e}")
                        live.update(render(state))
                        continue
                    except Exception as e:
                        with state_lock:
                            state["errors"].append(f"{fname}: {e}")
                        live.update(render(state))
                        continue

                    ul_exec.submit(do_upload, aid, fname, description)
                    dispatched += 1
                    live.update(render(state))
            except KeyboardInterrupt:
                interrupted = True
            finally:
                # Cancel any prefetched downloads we never consumed and drop
                # queued upload tasks. wait=True lets in-flight uploads finish
                # so the journal stays consistent with what's on the server.
                for _, _, fut in inflight:
                    fut.cancel()
                inflight.clear()
                dl_exec.shutdown(wait=True, cancel_futures=True)
                ul_exec.shutdown(wait=True, cancel_futures=True)
            live.update(render(state))

        if interrupted:
            console.print(
                f"\n[yellow]Interrupted.[/yellow] {state['run_count']} processed this run, "
                f"{len(state['errors'])} errors. Journal: {journal_path}",
                highlight=False,
            )
            sys.exit(130)

        console.print(
            f"\n[bold green]Done.[/bold green] {state['run_count']} processed this run, "
            f"{len(state['errors'])} errors. Journal: {journal_path}",
            highlight=False,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Catches a Ctrl+C that lands outside the work loop's handler
        # (e.g. during initial auth/setup) so we exit cleanly without a trace.
        sys.exit(130)

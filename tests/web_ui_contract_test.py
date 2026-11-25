#!/usr/bin/env python3
"""Offline contract and syntax checks for web/index.html.

Extracts the inline module script, parses it with `node --check` when node is
installed, and checks the wire-contract hooks the page must keep:

  * POST /v1/chat/completions built from the base URL, never hard-coded;
  * `session_id` in the request body and in the unified stream;
  * SSE reading through an async `readLines` generator;
  * `audio/pcm` playback that honours `sample_rate` and `channels`;
  * image, WAV and MP4 artifact rendering;
  * theme tokens that mirror Theme.swift;
  * every DOM id the script looks up exists in the markup;
  * Wave 10 interleaved narrative hooks (`visual_slot`, `/visual/render`,
    `narrative`, global streaming voice, in-place visual swap).

Usage: python3 -I tests/web_ui_contract_test.py
"""

import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "web" / "index.html"
THEME = REPO_ROOT / "clients" / "ummmlx" / "Sources" / "ummmlx" / "Theme.swift"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_page():
    require(PAGE.exists(), f"missing page: {PAGE}")
    return PAGE.read_text(encoding="utf-8")


def module_scripts(html):
    return re.findall(r'<script type="module">(.*?)</script>', html, re.S)


def check_markup(html):
    for tag in ('<main id="transcript">', 'id="prompt"', 'id="send"', 'id="stop"',
                'id="voiceToggle"', 'id="voiceWave"', 'id="voiceStatus"',
                'id="voiceMute"', 'id="voiceStop"', 'id="narrative"',
                'id="visualMode"', 'id="maxSlots"'):
        require(tag in html, f"missing element: {tag}")
    used = set(re.findall(r"\$\('([^']+)'\)", html))
    declared = set(re.findall(r'id="([^"]+)"', html))
    missing = sorted(used - declared)
    require(not missing, f"script looks up absent ids: {missing}")


def check_node(script):
    node = shutil.which("node")
    if not node:
        print("skip node --check: node is not installed", flush=True)
        return
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "page.mjs"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run([node, "--check", str(path)],
                                capture_output=True, text=True, timeout=60)
    require(result.returncode == 0,
            f"node --check failed: {result.stderr.strip()[:600]}")


def check_hooks(script):
    # The base URL is user input: the route is appended, /v1 is the default.
    require("'/chat/completions'" in script or '"/chat/completions"' in script,
            "the chat route must be appended to the configured base URL")
    require(re.search(r"if \(!path\) path = '/v1'", script),
            "an empty base path must default to /v1")
    require("method: 'POST'" in script, "the chat request must be a POST")
    require("session_id: sessionID" in script,
            "the request body must carry session_id")
    require("unified.session_id" in script,
            "the page must adopt the session_id the stream returns")

    require("async function* readLines" in script, "SSE must use an async readLines")
    require("res.body.getReader()" in script, "SSE must stream through getReader")
    require("line.startsWith('data:')" in script, "SSE must parse data lines")
    require("'[DONE]'" in script, "SSE must stop at [DONE]")
    require("unified.seq" in script, "the unified stream must be sequence-checked")

    require("'audio/pcm'" in script, "PCM audio must be played")
    require("chunk.sample_rate" in script, "PCM playback must honour sample_rate")
    require("chunk.channels" in script, "PCM playback must honour channels")
    require("createBuffer(" in script, "PCM must be scheduled as audio buffers")

    require("createElement('img')" in script, "image artifacts must render")
    require("createElement('audio')" in script, "WAV artifacts must render")
    require("createElement('video')" in script, "MP4 artifacts must render")
    require("'video/mp4'" in script, "the MP4 artifact type must be handled")
    require("body.top_p" in script and "body.seed" in script,
            "sampling controls top_p and seed must be wired")
    require("config.steps" in script and "config.image" in script,
            "image steps and edit image data URL must be wired")
    require("body.audio.ref_audio" in script,
            "audio voice-clone ref_audio data URL must be wired")
    require("config.num_frames" in script and "config.steps" in script
            and "config.guidance_scale" in script,
            "video fine-grained controls must be wired")


def check_wave10(script):
    # Global streaming voice bar.
    require("ummweb.voiceOn" in script,
            "the global Voice toggle must persist in localStorage")
    require("body.audio.stream = true" in script,
            "global voice must request audio.stream")
    require("player.setMuted" in script, "Mute must control a master gain")
    require("Stop Voice" in script, "the header must expose Stop Voice")

    # Narrative controls and request shape.
    require("body.narrative" in script, "the request must carry narrative")
    require("visual_mode" in script and "max_slots" in script,
            "narrative must send visual_mode and max_slots")
    require("ui.narrative.checked" in script, "the Narrative toggle must gate it")
    require("modalities.add('image')" in script and "modalities.add('video')" in script,
            "narrative must auto-select visual modalities")

    # Interleaved rendering.
    require("case 'visual_slot'" in script,
            "the stream must handle visual_slot events")
    require("unified.slot_id" in script, "slots must be attributed by slot_id")
    require("slot_index" in script or "slot-" in script,
            "visual slots need stable identities")
    require("turn.tailText" in script,
            "text must append to the trailing block, then resume below a slot")

    # In-place swap endpoint.
    require("'/visual/render'" in script or '"/visual/render"' in script,
            "visual swap must call /visual/render")
    require("'x-umminfer-kind'" in script or "'X-UmmInfer-Kind'" in script,
            "visual swap must honour the X-UmmInfer-Kind header")
    require("'→ Video'" in script and "'→ Image'" in script,
            "each slot must offer the image/video swap labels")
    require("'Regenerate'" in script and "'Download'" in script,
            "each slot must offer Regenerate and Download")
    require("slot.cache" in script,
            "swapping back must reuse cached media")


def check_wave11(script):
    # In-place true image-to-video animation: the slot keeps the generated
    # image as a data URL and forwards it when the video render runs.
    require("slot.imageDataURL" in script,
            "slots must retain the generated image as a data URL")
    require("dataURLFor(" in script and ";base64," in script,
            "image bytes must become a data:image/...;base64 URL")
    require("body.image = slot.imageDataURL" in script,
            "a video render must forward the slot image as body.image")
    require("'Animate → Video'" in script,
            "an image slot with a source frame must relabel the swap button")


def check_markdown(html, script):
    # Assistant bubbles render balanced Markdown via safe DOM nodes (no innerHTML).
    require("renderMarkdownInto(" in script,
            "assistant prose must render through renderMarkdownInto")
    require("safeLinkURL(" in script,
            "markdown links must be filtered through safeLinkURL")
    require(".innerHTML" not in script,
            "the web client must not assign innerHTML")
    require(".md h1 { font-size: 15.5px;" in html,
            "markdown h1 must stay compact and balanced (15.5px)")
    require(".md h2 { font-size: 14.5px;" in html,
            "markdown h2 must stay balanced (14.5px)")
    require(".md h3 { font-size: 13.5px;" in html,
            "markdown h3 must stay balanced (13.5px)")


def check_theme_tokens(html):
    """The CSS palette mirrors Theme.swift; both files must agree."""
    theme = THEME.read_text(encoding="utf-8")
    # Theme.swift builds the neutral tokens through gray(light, dark) and the
    # accent tokens through Color(hex: dark ? a : b). Both forms are pinned.
    pairs = [
        ("--bg", "background", 0xFFFFFF, 0x151515, "gray"),
        ("--surface", "surface", 0xF7F7F7, 0x1C1C1C, "gray"),
        ("--hover", "hover", 0xF2F2F2, 0x232323, "gray"),
        ("--line", "line", 0xEBEBEB, 0x2B2B2B, "gray"),
        ("--text", "text", 0x111111, 0xF2F2F2, "gray"),
        ("--secondary", "secondary", 0x6E6E6E, 0xA3A3A3, "gray"),
        ("--tertiary", "tertiary", 0xA3A3A3, 0x6B6B6B, "gray"),
        ("--ink", "ink", 0x111111, 0xF2F2F2, "gray"),
        ("--on-ink", "onInk", 0xFFFFFF, 0x151515, "gray"),
        ("--accent", "accent", 0x0D9384, 0x2BB3A3, "hex"),
        ("--danger", "danger", 0xEB5230, 0xFF6B47, "hex"),
    ]
    for name, member, light, dark, form in pairs:
        require(f"var({name})" in html, f"the page must use {name}")
        expected = (f"gray(0x{light:06X}, 0x{dark:06X})" if form == "gray"
                    else f"dark ? 0x{dark:06X} : 0x{light:06X}")
        require(expected in theme,
                f"Theme.{member} must be {expected}")
        require(f"{name}: #{light:06X}" in html,
                f"the light {name} must be #{light:06X}")
        require(f"{name}: #{dark:06X}" in html,
                f"the dark {name} must be #{dark:06X}")
    for metric in ("--r-control: 8px", "--r-card: 12px", "--f-title: 20px",
                   "--f-section: 15px", "--f-body: 13px", "--f-caption: 12px"):
        require(metric in html, f"the page must keep the metric {metric}")


def main():
    html = read_page()
    scripts = module_scripts(html)
    require(len(scripts) == 1, f"expected exactly one module script, found {len(scripts)}")
    script = scripts[0]
    checks = [
        ("markup", lambda: check_markup(html)),
        ("syntax", lambda: check_node(script)),
        ("hooks", lambda: check_hooks(script)),
        ("wave10", lambda: check_wave10(script)),
        ("wave11", lambda: check_wave11(script)),
        ("markdown", lambda: check_markdown(html, script)),
        ("theme", lambda: check_theme_tokens(html)),
    ]
    failures = 0
    for name, check in checks:
        try:
            check()
        except Exception as error:  # noqa: BLE001 - one report per check
            failures += 1
            print(f"FAIL {name}: {error}", flush=True)
        else:
            print(f"ok {name}", flush=True)
    if failures:
        print(f"{failures} of {len(checks)} checks failed", flush=True)
        return 1
    print(f"web UI contract passed ({len(script.splitlines())} script lines)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

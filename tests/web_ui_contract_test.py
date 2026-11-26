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
    `narrative`, global streaming voice, in-place visual swap);
  * opt-in realtime WS microphone/audio output, playback-consumption ACKs, and
    response.cancelled barge-in (including a mocked Web Audio behavior run).

Usage: python3 -I tests/web_ui_contract_test.py
"""

import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


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
    for tag in ('<main id="transcript">', 'id="emptyState"', 'id="showcasePresets"',
                'data-preset="documentary"', 'data-preset="storyboard_i2v"',
                'data-preset="smart_text_only"',
                'id="prompt"', 'id="send"', 'id="stop"',
                'id="voiceToggle"', 'id="voiceWave"', 'id="voiceStatus"',
                'id="voiceMute"', 'id="voiceStop"', 'id="realtimeToggle"',
                'id="realtimeEchoSuppression"', 'id="realtimeStatus"',
                'id="realtimeStrip"', 'id="realtimeStripPhase"',
                'id="realtimeEchoState"', 'id="realtimeBargeInCount"',
                'id="narrative"',
                'id="visualMode"', 'id="maxSlots"', 'id="allowTextOnly"',
                'id="narrativeContinuity"', 'id="styleAnchor"',
                'id="fastVideo"', 'id="videoFramePolicy"',
                'id="videoMotionThreshold"', 'id="videoSceneThreshold"'):
        require(tag in html, f"missing element: {tag}")
    require('<option value="off" selected>Off</option>' in html,
            "narrativeContinuity must default to Off")
    require('<option value="prompt_anchor">Prompt anchor</option>' in html
            and '<option value="i2v_chain">I2V chain</option>' in html,
            "narrativeContinuity must expose prompt_anchor and i2v_chain options")
    require('<input type="checkbox" id="fastVideo">' in html,
            "fastVideo checkbox must default to unchecked")
    require('<input type="checkbox" id="realtimeEchoSuppression" checked>' in html,
            "realtimeEchoSuppression checkbox must default to checked (ON)")
    require('<option value="uniform" selected>Uniform</option>' in html,
            "video frame selection must default to Uniform")
    for control in ('videoMotionThreshold', 'videoSceneThreshold'):
        require(re.search(rf'id="{control}"[^>]*\bdisabled\b', html) is not None,
                f"{control} must be disabled in default Uniform mode")
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
    # Opt-in Fast video (Wan2.2 TeaCache 0.05): default unchecked, persisted in
    # localStorage, disabled while running, forwarded on chat & visual/render.
    require("localStorage.getItem('ummweb.fastVideo') === '1'" in script,
            "fastVideo must default to unchecked unless persisted as '1'")
    require("localStorage.setItem('ummweb.fastVideo'" in script,
            "fastVideo must persist in localStorage")
    require("if (ui.fastVideo.checked) config.teacache_thresh = 0.05;" in script,
            "fastVideo must include teacache_thresh: 0.05 in chat body.video")
    require("if (ui.fastVideo.checked) body.teacache_thresh = 0.05;" in script,
            "fastVideo must include teacache_thresh: 0.05 in /visual/render video requests")
    require("ui.fastVideo" in script.split("function setBusy(busy)", 1)[1].split("}", 1)[0],
            "fastVideo must be disabled while running")


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
    require("allow_text_only: ui.allowTextOnly.checked" in script,
            "narrative must send allow_text_only from the allowTextOnly control")
    require("localStorage.getItem('ummweb.allowTextOnly') !== '0'" in script,
            "allowTextOnly must default to checked unless persisted as '0'")
    require("localStorage.setItem('ummweb.allowTextOnly'" in script,
            "allowTextOnly must persist to localStorage")
    require("syncNarrativeControls()" in script and "ui.allowTextOnly" in script,
            "allowTextOnly must be disabled alongside Narrative controls when Narrative is off or streaming")
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


def check_video_input(html, script):
    # Wave 13 native video input upload (+ Vid -> video_url content part).
    require('id="askVideo"' in html and 'id="pickVideo"' in html,
            "composer must expose askVideo file input and pickVideo button")
    require("type: 'video_url'" in script,
            "composer must serialize attached videos as video_url content parts")
    require("attachedVideo" in script,
            "composer must track attachedVideo state")
    require("videoURLPart(attachedVideo.dataURL, frameSelection)" in script,
            "attached video must use the shared frame-selection serializer")
    require("parseVideoFrameSelection(ui.videoFramePolicy.value" in script,
            "temporal settings must be validated while building the request")
    require("video.frame_selection = {" in script
            and "motion_threshold: frameSelection.motion_threshold" in script
            and "scene_threshold: frameSelection.scene_threshold" in script,
            "temporal mode must emit the accepted video_url.frame_selection schema")
    require("if (policy === 'uniform') return undefined;" in script,
            "Uniform must omit frame_selection to preserve legacy request bytes")
    for key in ('videoFramePolicy', 'videoMotionThreshold', 'videoSceneThreshold'):
        require(f"'{key}'" in script, f"{key} must persist with existing settings")
        require(f"ui.{key}" in script.split("function setBusy(busy)", 1)[1].split("])", 1)[0],
                f"{key} must be disabled while a request is running")
    require("syncVideoFrameControls()" in script,
            "threshold controls must synchronize with Uniform/Temporal and busy state")


def check_video_frame_selection_behavior(script):
    """Execute the exact request serializer without a browser or a server."""
    node = shutil.which("node")
    if not node:
        print("skip video frame-selection behavior: node is not installed", flush=True)
        return
    require("// VIDEO_FRAME_SELECTION_BEGIN" in script
            and "// VIDEO_FRAME_SELECTION_END" in script,
            "missing video frame-selection behavior-test markers")
    helpers = script.split("// VIDEO_FRAME_SELECTION_BEGIN", 1)[1].split(
        "// VIDEO_FRAME_SELECTION_END", 1)[0]
    video_config = re.search(r"function videoConfig\(\) \{.*?\n\}", script, re.S)
    require(video_config is not None, "could not extract videoConfig for interaction checks")
    harness = "const ui = {\n" + r"""
  videoSize: { value: '704x480' }, videoSeconds: { value: '2' },
  videoFrames: { value: '29' }, videoSteps: { value: '25' },
  videoGuidance: { value: '5.0' }, fastVideo: { checked: false },
};
""" + video_config.group(0) + helpers + r"""
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const dataURL = 'data:video/mp4;base64,AAAA';
const legacy = '{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}';

// Default and reset-to-Uniform are byte-identical to the pre-feature part.
assert(JSON.stringify(videoURLPart(dataURL, undefined)) === legacy,
       'default video_url bytes changed');
assert(parseVideoFrameSelection('uniform', 'not-a-number', '') === undefined,
       'Uniform must ignore disabled temporal fields');
assert(JSON.stringify(videoURLPart(
  dataURL, parseVideoFrameSelection('uniform', '0.02', '0.25'))) === legacy,
       'reset to Uniform did not restore the legacy request');

// Temporal opt-in emits exactly the accepted schema, including 0/1 bounds.
const temporal = parseVideoFrameSelection('temporal', '0.02', '0.25');
assert(JSON.stringify(videoURLPart(dataURL, temporal)) ===
  '{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA","frame_selection":{"policy":"temporal","motion_threshold":0.02,"scene_threshold":0.25}}}',
  'temporal wire schema mismatch');
const boundaries = videoURLPart(
  dataURL, parseVideoFrameSelection('temporal', '0', '1'));
assert(boundaries.video_url.frame_selection.motion_threshold === 0 &&
       boundaries.video_url.frame_selection.scene_threshold === 1,
       'inclusive 0/1 boundaries were rejected');

// Every malformed, non-finite, or out-of-range temporal value fails before fetch.
const invalid = ['', ' ', 'abc', '0.2x', 'NaN', 'Infinity', '-Infinity',
                 '-0.0001', '1.0001', '-1', '2'];
for (const value of invalid) {
  for (const field of ['motion', 'scene']) {
    const motion = field === 'motion' ? value : '0.02';
    const scene = field === 'scene' ? value : '0.25';
    let rejected = false;
    try { parseVideoFrameSelection('temporal', motion, scene); }
    catch (error) {
      rejected = /finite number in \[0,1\]/.test(error.message);
    }
    assert(rejected, `invalid ${field} threshold was accepted: ${value}`);
  }
}
let badPolicyRejected = false;
try { parseVideoFrameSelection('scene-only', '0.02', '0.25'); }
catch { badPolicyRejected = true; }
assert(badPolicyRejected, 'unknown frame policy was accepted');

// A reconnect reads fresh values; no failed or previous request state leaks.
let reconnectRejected = false;
try { parseVideoFrameSelection('temporal', 'oops', '0.25'); }
catch { reconnectRejected = true; }
assert(reconnectRejected, 'invalid pre-reconnect value was not rejected');
const reconnected = parseVideoFrameSelection('temporal', '1', '0');
assert(reconnected.motion_threshold === 1 && reconnected.scene_threshold === 0 &&
       temporal.motion_threshold === 0.02 && temporal.scene_threshold === 0.25,
       'reconnect reused or mutated stale temporal settings');

// Existing generated-video controls and opt-in Fast Video stay independent.
const normalGeneration = videoConfig();
assert(normalGeneration.size === '704x480' && normalGeneration.num_frames === 29 &&
       normalGeneration.steps === 25 && normalGeneration.guidance_scale === 5 &&
       normalGeneration.teacache_thresh === undefined,
       'existing generated-video defaults changed');
ui.fastVideo.checked = true;
const fastGeneration = videoConfig();
assert(fastGeneration.teacache_thresh === 0.05 &&
       fastGeneration.num_frames === 29 && fastGeneration.steps === 25,
       'Fast Video interaction regressed');
assert(videoURLPart(dataURL, reconnected).video_url.frame_selection.policy === 'temporal',
       'Fast Video altered video_url.frame_selection');
console.log('video frame-selection behavior passed');
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "video-frame-selection.mjs"
        path.write_text(harness, encoding="utf-8")
        result = subprocess.run([node, str(path)], capture_output=True,
                                text=True, timeout=60)
    require(result.returncode == 0,
            f"video frame-selection behavior failed: {(result.stderr or result.stdout).strip()[:1200]}")
    require("video frame-selection behavior passed" in result.stdout,
            "video frame-selection harness did not reach its success assertion")


def check_narrative_continuity(html, script):
    """Stage B cross-slot continuity controls, persistence, and attribution hooks."""
    require('id="narrativeContinuity"' in html and 'id="styleAnchor"' in html,
            "narrative continuity controls must exist in markup")
    for key in ('narrativeContinuity', 'styleAnchor'):
        require(f"'{key}'" in script,
                f"{key} must persist in localStorage via SETTINGS")
        require(f"ui.{key}" in script.split("function setBusy(busy)", 1)[1].split("])", 1)[0],
                f"{key} must be disabled while streaming")
        require(f"ui.{key}" in script.split("function syncNarrativeControls()", 1)[1].split("}", 1)[0],
                f"{key} must be synchronized with narrative toggle and busy state")
    require("buildNarrativeConfig(" in script,
            "send() must build body.narrative through buildNarrativeConfig")
    require("new TextEncoder().encode(trimmed).length" in script,
            "style_anchor must enforce UTF-8 byte length via TextEncoder")
    require("visualSlotContinuityBadge(" in script,
            "visual_slot cards must render continuity attribution badges")
    require("anchor_explicit" in script and "anchor_sha256" in script,
            "SSE handling must parse optional anchor_explicit and anchor_sha256 metadata")
    require("Explicit I2V ←" in script,
            "explicit anchor badges must render Explicit I2V label")
    require("updateSlotContinuity(" in script and "Array.isArray(unified.legs)" in script,
            "slot continuity must update from artifact, leg_error, and complete.legs")

    send_fn = script.split("async function send()", 1)[1]
    require(send_fn.index("buildNarrativeConfig(") < send_fn.index("attachedImage = null;"),
            "style_anchor validation must run before clearing attachments")
    require(send_fn.index("buildNarrativeConfig(") < send_fn.index("ui.prompt.value = '';"),
            "style_anchor validation must run before clearing the prompt input")
    require(send_fn.index("buildNarrativeConfig(") < send_fn.index("await fetch("),
            "style_anchor validation must abort before fetch")


def check_narrative_continuity_behavior(script):
    """Execute narrative continuity serialization, UTF-8 validation, and slot badge DOM in Node."""
    node = shutil.which("node")
    if not node:
        print("skip narrative continuity behavior: node is not installed", flush=True)
        return
    require("// NARRATIVE_CONTINUITY_BEGIN" in script
            and "// NARRATIVE_CONTINUITY_END" in script,
            "missing narrative continuity behavior-test markers")
    helpers = script.split("// NARRATIVE_CONTINUITY_BEGIN", 1)[1].split(
        "// NARRATIVE_CONTINUITY_END", 1)[0]
    create_slot = re.search(r"function createSlot\(.*?\n\}", script, re.S)
    show_loading = re.search(r"function showSlotLoading\(slot\) \{.*?\n\}", script, re.S)
    show_error = re.search(r"function showSlotError\(slot, message\) \{.*?\n\}", script, re.S)
    require(create_slot is not None and show_loading is not None and show_error is not None,
            "could not extract createSlot/showSlotLoading/showSlotError for badge behavior checks")
    harness = helpers + "\n" + show_loading.group(0) + "\n" + show_error.group(0) + "\n" + create_slot.group(0) + r"""
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

// 1. Default 'off' + empty/whitespace omits continuity_mode and style_anchor completely.
const defaultNav = buildNarrativeConfig({
  visual_mode: 'auto', max_slots: '3', allow_text_only: true,
  continuity_mode: 'off', style_anchor: '',
});
assert(JSON.stringify(defaultNav) ===
  '{"enabled":true,"visual_mode":"auto","max_slots":3,"allow_text_only":true}',
  'default off + empty must emit zero extra keys in narrative');
const whitespaceNav = buildNarrativeConfig({
  visual_mode: 'auto', max_slots: '3', allow_text_only: true,
  continuity_mode: 'off', style_anchor: '   \t\n  ',
});
assert(JSON.stringify(whitespaceNav) === JSON.stringify(defaultNav),
  'whitespace-only style_anchor must be omitted');

// 2. i2v_chain + valid style_anchor serializes both accurately and trims whitespace.
const chainedNav = buildNarrativeConfig({
  visual_mode: 'auto', max_slots: '4', allow_text_only: false,
  continuity_mode: 'i2v_chain', style_anchor: '  warm watercolor ink  ',
});
assert(JSON.stringify(chainedNav) ===
  '{"enabled":true,"visual_mode":"auto","max_slots":4,"allow_text_only":false,"continuity_mode":"i2v_chain","style_anchor":"warm watercolor ink"}',
  'i2v_chain + style_anchor wire shape mismatch');

// 3. Exact 256 UTF-8 byte boundary with multi-byte Chinese characters succeeds.
const cjk256 = '墨'.repeat(85) + 'a'; // 85 * 3 + 1 = 256 UTF-8 bytes
assert(new TextEncoder().encode(cjk256).length === 256, 'expected 256 UTF-8 bytes');
const exactNav = buildNarrativeConfig({
  visual_mode: 'auto', max_slots: '3', allow_text_only: true,
  continuity_mode: 'prompt_anchor', style_anchor: cjk256,
});
assert(exactNav.continuity_mode === 'prompt_anchor' && exactNav.style_anchor === cjk256,
  'exact 256-byte multi-byte style_anchor must be accepted');

// 4. Oversized UTF-8 style_anchor (> 256 bytes, ASCII and multi-byte Chinese) is rejected pre-fetch.
for (const oversized of ['a'.repeat(257), '墨'.repeat(86)]) {
  assert(new TextEncoder().encode(oversized).length > 256, 'test input must exceed 256 UTF-8 bytes');
  let rejected = false;
  try {
    buildNarrativeConfig({
      visual_mode: 'auto', max_slots: '3', allow_text_only: true,
      continuity_mode: 'i2v_chain', style_anchor: oversized,
    });
  } catch (err) {
    rejected = /256 UTF-8 bytes/.test(err.message);
  }
  assert(rejected, 'oversized style_anchor was not rejected');
}

// 5. Invalid continuity_mode is rejected.
let badMode = false;
try {
  buildNarrativeConfig({ continuity_mode: 'magic_chain', style_anchor: '' });
} catch { badMode = true; }
assert(badMode, 'unknown continuity_mode was accepted');

// 6. visual_slot continuity badges render cleanly in DOM.
class FakeElement {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.className = '';
    this.textContent = '';
    this.title = '';
    this.dataset = {};
    this.children = [];
    this.hidden = false;
    this._classes = new Set();
    this.classList = {
      add: (...names) => names.forEach((n) => this._classes.add(n)),
      remove: (...names) => names.forEach((n) => this._classes.delete(n)),
      contains: (n) => this._classes.has(n),
    };
  }
  append(...nodes) { this.children.push(...nodes); }
  insertBefore(node, ref) {
    const idx = this.children.indexOf(ref);
    if (idx >= 0) this.children.splice(idx, 0, node);
    else this.children.push(node);
  }
  replaceChildren(...nodes) { this.children = [...nodes]; }
  addEventListener() {}
}
globalThis.document = { createElement: (tag) => new FakeElement(tag) };

const sampleSha = '3a9d5bbe' + '4f12c890'.repeat(7);
assert(sampleSha.length === 64, 'sampleSha must be 64 hex chars');

const chainBadge = visualSlotContinuityBadge({
  continuity_mode: 'i2v_chain', anchor_slot_index: 0,
});
assert(chainBadge && chainBadge.text === 'I2V ← Slot 1' && chainBadge.kind === 'i2v-chain' &&
       chainBadge.explicit === false && chainBadge.sha256 === null,
  'i2v_chain anchor_slot_index 0 must map to I2V ← Slot 1');
const chainSlot = createSlot('slot-1', 'lantern drifting in rain', 'video', chainBadge);
const headChildren = chainSlot.el.children[0].children;
assert(headChildren.some((node) => node.textContent === 'I2V ← Slot 1' &&
       node.dataset.continuity === 'i2v-chain'),
  'createSlot did not render the I2V ← Slot 1 badge');

// Implicit I2V with anchor_sha256 preserves implicit label and adds SHA-256 hover tooltip.
const implicitWithSha = visualSlotContinuityBadge({
  continuity_mode: 'i2v_chain', anchor_slot_index: 0, anchor_explicit: false,
  anchor_sha256: sampleSha.toUpperCase(),
});
assert(implicitWithSha && implicitWithSha.text === 'I2V ← Slot 1' &&
       implicitWithSha.explicit === false && implicitWithSha.sha256 === sampleSha &&
       implicitWithSha.title.includes(sampleSha),
  'implicit i2v_chain with sha256 must preserve label and include full digest tooltip');

// Explicit source=slot-0 with 64-char anchor_sha256 renders Explicit I2V + 8-char digest + tooltip.
const explicitWithSha = visualSlotContinuityBadge({
  continuity_mode: 'i2v_chain', anchor_slot_index: 0,
  anchor_explicit: true, anchor_sha256: sampleSha,
});
assert(explicitWithSha && explicitWithSha.text === 'Explicit I2V ← Slot 1 · 3a9d5bbe' &&
       explicitWithSha.kind === 'i2v-chain' && explicitWithSha.explicit === true &&
       explicitWithSha.sha256 === sampleSha && explicitWithSha.title.includes(sampleSha),
  'explicit anchor with sha256 must render Explicit I2V ← Slot 1 · 3a9d5bbe + full tooltip');
const explicitSlot = createSlot('slot-1', 'lantern drifting', 'video', explicitWithSha);
const explicitNode = explicitSlot.el.children[0].children.find(
  (node) => node.dataset.continuity === 'i2v-chain');
assert(explicitNode && explicitNode.textContent === 'Explicit I2V ← Slot 1 · 3a9d5bbe' &&
       explicitNode.dataset.anchorExplicit === 'true' &&
       explicitNode.dataset.anchorSha256 === sampleSha &&
       explicitNode.title.includes(sampleSha),
  'createSlot did not render explicit anchor badge text, datasets, and tooltip');

// Explicit source=slot-0 without sha256 (or with null/empty/malformed sha256) omits digest cleanly.
for (const absent of [undefined, null, '', 'not-a-sha', '3a9d5bbe']) {
  const explicitNoSha = visualSlotContinuityBadge({
    continuity_mode: 'i2v_chain', anchor_slot_index: 0,
    anchor_explicit: true, anchor_sha256: absent,
  });
  assert(explicitNoSha && explicitNoSha.text === 'Explicit I2V ← Slot 1' &&
         explicitNoSha.explicit === true && explicitNoSha.sha256 === null,
    `explicit anchor without valid 64-char sha256 must omit short digest (got ${explicitNoSha?.text})`);
}

// Late arrival of anchor_sha256 (via artifact or complete.legs) upgrades badge in-place.
const pendingExplicit = createSlot(
  'slot-1', 'lantern drifting', 'video',
  visualSlotContinuityBadge({ continuity_mode: 'i2v_chain', anchor_slot_index: 0, anchor_explicit: true }),
);
pendingExplicit.continuityMeta = {
  continuity_mode: 'i2v_chain', anchor_slot_index: 0, anchor_explicit: true, anchor_sha256: '',
};
assert(pendingExplicit.continuityEl.textContent === 'Explicit I2V ← Slot 1',
  'initial explicit slot badge before digest resolved mismatch');
updateSlotContinuity(pendingExplicit, {
  slot_id: 'slot-1', anchor_slot_index: 0, anchor_explicit: true, anchor_sha256: sampleSha,
});
assert(pendingExplicit.continuityEl.textContent === 'Explicit I2V ← Slot 1 · 3a9d5bbe' &&
       pendingExplicit.continuityEl.title.includes(sampleSha) &&
       pendingExplicit.continuityEl.dataset.anchorSha256 === sampleSha,
  'updateSlotContinuity did not upgrade explicit badge with resolved anchor_sha256');

// Explicit slot leg_error renders error message cleanly inside slot card.
showSlotError(pendingExplicit, 'source slot slot-1 is not an earlier slot');
assert(pendingExplicit.hasError === true && pendingExplicit.el.dataset.error === 'true' &&
       pendingExplicit.el.classList.contains('has-error') &&
       pendingExplicit.media.children[0].children[0].textContent.includes('source slot slot-1 is not an earlier slot'),
  'showSlotError did not render explicit leg_error message on the slot card');

const anchorBadge = visualSlotContinuityBadge({ continuity_mode: 'prompt_anchor' });
assert(anchorBadge && anchorBadge.text === 'Anchor' && anchorBadge.kind === 'prompt-anchor',
  'prompt_anchor must map to Anchor badge');
const plainBadge = visualSlotContinuityBadge({ slot_id: 'slot-0', media_type: 'image' });
assert(plainBadge === null, 'plain visual_slot must not emit a continuity badge');
const plainSlot = createSlot('slot-0', 'still lake', 'image', plainBadge);
assert(plainSlot.el.children[0].children.length === 2,
  'plain slot header should have only kind badge and caption');

console.log('narrative continuity behavior passed');
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "narrative-continuity.mjs"
        path.write_text(harness, encoding="utf-8")
        result = subprocess.run([node, str(path)], capture_output=True,
                                text=True, timeout=60)
    require(result.returncode == 0,
            f"narrative continuity behavior failed: {(result.stderr or result.stdout).strip()[:1200]}")
    require("narrative continuity behavior passed" in result.stdout,
            "narrative continuity harness did not reach its success assertion")


def check_realtime(html, script):
    """Stage B realtime wire hooks stay opt-in and independent of HTTP/SSE."""
    require(re.search(
        r'id="realtimeToggle"[^>]*aria-pressed="false"', html),
        "realtime microphone/WS mode must be default-off and opt-in")
    require("ui.realtimeToggle.addEventListener('click'" in script,
            "the realtime toggle must be wired")
    require("else startRealtime()" in script,
            "the opt-in toggle must start the realtime session")
    require("'/realtime'" in script and "new WebSocket(url)" in script,
            "realtime must connect a WebSocket to the configured /realtime route")
    require("url.protocol === 'https:' ? 'wss:' : 'ws:'" in script,
            "realtime WS security must follow the configured HTTP endpoint")
    require("ui.realtimeEndpoint.value.trim() || ui.endpoint.value" in script,
            "an independent --realtime-port listener must be configurable")
    require("Browser Realtime WS cannot send the configured API key" in script,
            "unsupported browser API-key handshakes must fail fast without URL credentials")
    require("intent: 'transcription'" in script and "model }).toString()" in script,
            "the realtime handshake must carry intent=transcription and model")

    # Session negotiation and full-duplex microphone input.
    require("type: 'session.update'" in script,
            "realtime must negotiate session.update")
    require("buildRealtimeInitialSessionUpdate(" in script and
            "buildRealtimeEchoSuppressionUpdate(" in script and
            "parseRealtimeSessionUpdated(" in script and
            "sendRealtimeEchoSuppressionToggle(" in script,
            "realtime must wire echo-suppression session.update helpers")
    require("ui.realtimeEchoSuppression.addEventListener('change'" in script,
            "realtime echo suppression toggle must listen for live changes")
    require("ui.realtimeEchoSuppression" not in
            script.split("function setBusy(busy)", 1)[1].split("}", 1)[0],
            "realtime echo suppression toggle must stay usable without setBusy disabling it")
    require("recordRealtimeBargeIn(realtime);" in script,
            "response.cancelled must increment the observed barge-in counter")
    require("resetRealtimeTelemetry(realtime);" in script,
            "disconnect/reconnect must reset realtime echo and barge-in telemetry")
    require("Echo suppression: On" in script and "Echo suppression: Off (A/B)" in script,
            "live voice status strip must format confirmed echo suppression states")
    require("input_audio_format: 'pcm16'" in script and
            "type: 'server_vad'" in script,
            "realtime must select PCM16 input and server VAD")
    require(re.search(r"audio_output:\s*\{.*?playback_ack: true", script, re.S),
            "audio_output.playback_ack=true must be negotiated")
    require("event.session?.audio_output?.playback_ack !== true" in script,
            "the accepted session must be checked for playback_ack")
    require("getUserMedia" in script and "createMediaStreamSource" in script,
            "realtime must capture microphone input")
    require("resampleMono(input, ctx.sampleRate, 16000)" in script and
            "encodePcm16Mono(pcm16k)" in script,
            "microphone frames must be converted to 16 kHz PCM16")
    require("type: 'input_audio_buffer.append'" in script and
            "audio: encodeBase64(pcmBytes)" in script,
            "16 kHz PCM16 mic frames must use input_audio_buffer.append")
    require("if (!handedOff)" in script and
            "stream?.getTracks().forEach((track) => track.stop())" in script,
            "microphone tracks must be released if graph setup fails after permission")

    # PCM playback and consumption-based monotonic acknowledgement.
    require("case 'audio.delta'" in script and "realtimePlayback.push(" in script,
            "audio.delta must feed the realtime PCM player")
    require("source.onended = () =>" in script,
            "playback ACK must be driven by actual buffer completion")
    require("this.acknowledge(responseId, textCursor, false)" in script and
            "textCursor <= this.lastAckCursor" in script,
            "delta ACK cursors must be monotonic and acknowledged after consumption")
    require("this.acknowledgeFinalIfDrained()" in script and
            "this.sources.size !== 0" in script,
            "the final ACK must wait until every scheduled source is consumed")
    require("type: 'output_audio_buffer.playback_ack'" in script and
            "response_id: responseId" in script and "text_cursor: textCursor" in script,
            "playback ACK must carry response_id and text_cursor")
    require("case 'audio.done'" in script and "realtimePlayback.done(" in script,
            "audio.done must arm the final playback ACK")

    # Barge-in: invalidate first, stop scheduled sources, then drop late frames.
    cancellation = script.split("function cancelRealtimeResponse(", 1)[1].split("\n}", 1)[0]
    require(cancellation.index("markRealtimeStale(id)") <
            cancellation.index("realtimePlayback.cancel(id)"),
            "response cancellation must mark stale before stopping playback")
    require("responseId || realtime.activeResponseId || realtimePlayback.responseId" in
            cancellation,
            "Stop Voice and id-less failures must fall back to the active response")
    cancelled = script.split("case 'response.cancelled':", 1)[1].split("break;", 1)[0]
    require("cancelRealtimeResponse(responseId)" in cancelled,
            "response.cancelled must use the common immediate cancellation path")
    failed = script.split("case 'response.failed':", 1)[1].split("break;", 1)[0]
    require("cancelRealtimeResponse(responseId)" in failed,
            "an id-less response.failed must cancel active playback and ACKs")
    voice_stop = script.split("ui.voiceStop.addEventListener", 1)[1].split("});", 1)[0]
    require("cancelRealtimeResponse();" in voice_stop,
            "Stop Voice must suppress even a text-only response before its first audio")
    require("record.source.stop(0)" in script,
            "barge-in must immediately stop scheduled audio sources")
    require("realtime.cancelledResponses.has(responseId)" in script and
            "this.cancelled.has(responseId)" in script,
            "late events and onended callbacks must drop cancelled response ids")
    require("if (!realtime.enabled || !realtime.ready" in script,
            "microphone frames must be gated by negotiated realtime readiness")

    # The existing HTTP/SSE implementation is still separate and present.
    require("const player = {" in script and "async function send()" in script and
            "for await (const line of readLines(res))" in script,
            "the existing HTTP/SSE chat and narrative path must remain intact")


def check_realtime_behavior(script):
    """Execute the player with controllable Web Audio nodes, without a server."""
    node = shutil.which("node")
    if not node:
        print("skip realtime behavior: node is not installed", flush=True)
        return
    pcm = re.search(
        r"function pcmBuffer\(ctx, bytes, rate, channels\) \{.*?\n\}", script, re.S)
    require(pcm is not None, "could not extract pcmBuffer for the behavior test")
    require("// REALTIME_PLAYER_CLASS_BEGIN" in script and
            "// REALTIME_PLAYER_CLASS_END" in script,
            "missing realtime player behavior-test markers")
    player = script.split("// REALTIME_PLAYER_CLASS_BEGIN", 1)[1].split(
        "// REALTIME_PLAYER_CLASS_END", 1)[0]
    harness = pcm.group(0) + player + r"""
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
class FakeAudioContext {
  constructor() {
    this.currentTime = 10;
    this.state = 'running';
    this.destination = {};
    this.sampleRate = 24000;
  }
  createGain() {
    return { gain: { value: 1 }, connect() {}, disconnect() {} };
  }
  createBuffer(channels, frames, rate) {
    return {
      duration: frames / rate,
      getChannelData() { return new Float32Array(frames); },
    };
  }
  createBufferSource() {
    return {
      buffer: null,
      onended: null,
      stopped: false,
      started: false,
      connect() {},
      disconnect() {},
      start() { this.started = true; },
      stop() { this.stopped = true; },
    };
  }
  resume() { this.state = 'running'; return Promise.resolve(); }
  close() { this.state = 'closed'; return Promise.resolve(); }
}
globalThis.AudioContext = FakeAudioContext;
const acks = [];
const player = new RealtimePCMPlayer(
  (responseId, textCursor) => acks.push({ responseId, textCursor }),
  () => {},
);
await player.unlock();
const pcmBytes = new Uint8Array(480 * 2);
const cursors = () => acks.map((ack) => ack.textCursor);

// Receipt, scheduling, and audio.done do not acknowledge unplayed audio.
player.push('r1', pcmBytes, 24000, 1, 0);
player.push('r1', pcmBytes, 24000, 1, 12);
player.done('r1', 24);
assert(acks.length === 0, 'audio was acknowledged before playback consumption');
const firstSources = [...player.sources].map((record) => record.source);
firstSources[0].onended();
assert(JSON.stringify(cursors()) === '[0]',
       'only the consumed first chunk cursor should be acknowledged');
firstSources[1].onended();
assert(JSON.stringify(cursors()) === '[0,12,24]',
       'last delta and final cursor must follow playback drain in order');

// Repeated intermediate cursors are suppressed. The equal final cursor is
// allowed exactly once after audio.done so the server can commit the turn.
player.push('r2', pcmBytes, 24000, 1, 5);
player.push('r2', pcmBytes, 24000, 1, 5);
const repeatedSources = [...player.sources].map((record) => record.source);
const beforeRepeated = acks.length;
repeatedSources[0].onended();
repeatedSources[1].onended();
assert(acks.length === beforeRepeated + 1 && acks.at(-1).textCursor === 5,
       'repeated intermediate cursor must not be acknowledged twice');
player.done('r2', 5);
assert(acks.length === beforeRepeated + 2 && acks.at(-1).textCursor === 5,
       'equal final cursor must be acknowledged once after audio.done');
player.done('r2', 5);
assert(acks.length === beforeRepeated + 2,
       'repeated audio.done must not repeat the final ACK');

// Cancellation stops every scheduled source immediately. Even a captured
// stale onended callback, a late delta, or audio.done cannot emit another ACK.
player.push('r3', pcmBytes, 24000, 1, 7);
player.push('r3', pcmBytes, 24000, 1, 9);
const cancelledRecords = [...player.sources];
const staleEnded = cancelledRecords.map((record) => record.source.onended);
const beforeCancel = acks.length;
player.cancel('r3');
assert(cancelledRecords.every((record) => record.source.stopped),
       'response cancellation did not stop every scheduled source');
staleEnded.forEach((callback) => callback());
player.push('r3', pcmBytes, 24000, 1, 11);
player.done('r3', 13);
assert(acks.length === beforeCancel,
       'stale response frames or callbacks acknowledged cancelled playback');

// A new response has its own cursor domain and cannot release old callbacks.
player.push('r4', pcmBytes, 24000, 1, 2);
const currentSource = [...player.sources][0].source;
currentSource.onended();
assert(acks.at(-1).responseId === 'r4' && acks.at(-1).textCursor === 2,
       'new response playback did not acknowledge its own cursor');
player.stop();
console.log('realtime playback behavior passed');
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "realtime-player.mjs"
        path.write_text(harness, encoding="utf-8")
        result = subprocess.run([node, str(path)], capture_output=True,
                                text=True, timeout=60)
    require(result.returncode == 0,
            f"realtime behavior failed: {(result.stderr or result.stdout).strip()[:1000]}")
    require("realtime playback behavior passed" in result.stdout,
            "realtime behavior harness did not reach its success assertion")


def check_realtime_echo_suppression_behavior(script):
    """Execute realtime echo-suppression wire helpers and barge-in status formatting in Node."""
    node = shutil.which("node")
    if not node:
        print("skip realtime echo-suppression behavior: node is not installed", flush=True)
        return
    require("// REALTIME_ECHO_SUPPRESSION_BEGIN" in script and
            "// REALTIME_ECHO_SUPPRESSION_END" in script,
            "missing realtime echo-suppression behavior-test markers")
    helpers = script.split("// REALTIME_ECHO_SUPPRESSION_BEGIN", 1)[1].split(
        "// REALTIME_ECHO_SUPPRESSION_END", 1)[0]
    harness = helpers + r"""
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

// 1. Default initial session.update (echoSuppression=true) OMITS echo_suppression.
const defaultInit = buildRealtimeInitialSessionUpdate({
  voice: 'vivian', maxTokens: 256, silenceMs: 400, echoSuppression: true,
});
assert(defaultInit.type === 'session.update', 'initial type mismatch');
assert(JSON.stringify(defaultInit.session.turn_detection) ===
  '{"type":"server_vad","silence_ms":400}',
  'default echoSuppression=true must omit echo_suppression on the wire: ' +
  JSON.stringify(defaultInit.session.turn_detection));
assert(!Object.prototype.hasOwnProperty.call(defaultInit.session.turn_detection, 'echo_suppression'),
  'echo_suppression key must be absent when default true');
assert(defaultInit.session.input_audio_format === 'pcm16' &&
  defaultInit.session.audio_output.playback_ack === true &&
  defaultInit.session.audio_output.voice === 'vivian' &&
  defaultInit.session.max_new_tokens === 256,
  'initial session.update base fields regressed');

// 2. Opt-out initial session.update (echoSuppression=false) emits explicit false.
const optOutInit = buildRealtimeInitialSessionUpdate({
  voice: 'vivian', maxTokens: 256, silenceMs: 400, echoSuppression: false,
});
assert(JSON.stringify(optOutInit.session.turn_detection) ===
  '{"type":"server_vad","silence_ms":400,"echo_suppression":false}',
  'opt-out echoSuppression=false must include echo_suppression:false');

// 3. Narrow active mid-session toggle sends ONLY turn_detection (no format/audio/max_tokens).
const narrowOff = buildRealtimeEchoSuppressionUpdate(false, 600);
assert(narrowOff.type === 'session.update', 'narrow update type mismatch');
assert(JSON.stringify(Object.keys(narrowOff.session)) === '["turn_detection"]',
  'narrow active update must contain ONLY turn_detection, got ' +
  JSON.stringify(Object.keys(narrowOff.session)));
assert(JSON.stringify(narrowOff.session.turn_detection) ===
  '{"type":"server_vad","silence_ms":600,"echo_suppression":false}',
  'narrow active OFF payload mismatch');

const narrowOn = buildRealtimeEchoSuppressionUpdate(true, 600);
assert(JSON.stringify(Object.keys(narrowOn.session)) === '["turn_detection"]',
  'narrow active ON update must contain ONLY turn_detection');
assert(JSON.stringify(narrowOn.session.turn_detection) ===
  '{"type":"server_vad","silence_ms":600,"echo_suppression":true}',
  'narrow active ON payload mismatch');

// 4. sendRealtimeEchoSuppressionToggle sends only on an OPEN socket.
const sentFrames = [];
const closedSocket = { readyState: 0, send(frame) { sentFrames.push(frame); } };
assert(sendRealtimeEchoSuppressionToggle(false, 400, closedSocket) === null,
  'must not send toggle on non-open socket');
assert(sentFrames.length === 0, 'closed socket received unexpected frame');

const openSocket = { readyState: 1, send(frame) { sentFrames.push(frame); } };
const sentPayload = sendRealtimeEchoSuppressionToggle(false, 450, openSocket);
assert(sentPayload && sentFrames.length === 1, 'open socket did not receive narrow toggle');
assert(sentFrames[0] ===
  '{"type":"session.update","session":{"turn_detection":{"type":"server_vad","silence_ms":450,"echo_suppression":false}}}',
  'sent narrow toggle wire bytes mismatch: ' + sentFrames[0]);

// 5. parseRealtimeSessionUpdated handles omitted=>true, false=>false, true=>true, and rejects bad types.
const parsedOmitted = parseRealtimeSessionUpdated({
  type: 'transcription_session.updated',
  session: {
    input_audio_format: 'pcm16',
    turn_detection: { type: 'server_vad', silence_ms: 400 },
    audio_output: { voice: 'vivian', playback_ack: true },
  },
});
assert(parsedOmitted.echoSuppression === true && parsedOmitted.silenceMs === 400 &&
  parsedOmitted.playbackAck === true,
  'omitted echo_suppression must parse as true');

const parsedFalse = parseRealtimeSessionUpdated({
  type: 'transcription_session.updated',
  session: {
    turn_detection: { type: 'server_vad', silence_ms: 550, echo_suppression: false },
    audio_output: { voice: 'vivian', playback_ack: true },
  },
});
assert(parsedFalse.echoSuppression === false && parsedFalse.silenceMs === 550,
  'explicit false echo_suppression must parse as false');

const parsedTrue = parseRealtimeSessionUpdated({
  type: 'transcription_session.updated',
  session: {
    turn_detection: { type: 'server_vad', silence_ms: 400, echo_suppression: true },
    audio_output: { voice: 'vivian', playback_ack: true },
  },
});
assert(parsedTrue.echoSuppression === true,
  'explicit true echo_suppression must parse as true');

let badEchoRejected = false;
try {
  parseRealtimeSessionUpdated({
    type: 'transcription_session.updated',
    session: { turn_detection: { type: 'server_vad', silence_ms: 400, echo_suppression: 'no' } },
  });
} catch { badEchoRejected = true; }
assert(badEchoRejected, 'non-boolean echo_suppression must be rejected');

// 6. Status strip labels (zero fabrication when unconfirmed).
assert(formatRealtimeEchoSuppressionLabel(null) === '',
  'unconfirmed echo suppression must render empty string');
assert(formatRealtimeEchoSuppressionLabel(undefined) === '',
  'undefined echo suppression must render empty string');
assert(formatRealtimeEchoSuppressionLabel(true) === 'Echo suppression: On',
  'confirmed true label mismatch');
assert(formatRealtimeEchoSuppressionLabel(false) === 'Echo suppression: Off (A/B)',
  'confirmed false label mismatch');

// 7. Barge-in counter increment, formatting, and disconnect/reconnect reset semantics.
const sessionState = resetRealtimeTelemetry({});
assert(sessionState.negotiatedSilenceMs === 400 &&
  sessionState.confirmedEchoSuppression === null &&
  sessionState.bargeInCount === 0,
  'fresh telemetry state must start unconfirmed with zero barge-ins');
assert(formatRealtimeBargeInLabel(sessionState.bargeInCount) === 'Barge-ins: 0',
  'zero barge-in label mismatch');
assert(recordRealtimeBargeIn(sessionState) === 1, 'first barge-in must increment to 1');
assert(recordRealtimeBargeIn(sessionState) === 2, 'second barge-in must increment to 2');
assert(formatRealtimeBargeInLabel(sessionState.bargeInCount) === 'Barge-ins: 2',
  'observed barge-in label mismatch');
sessionState.confirmedEchoSuppression = false;
sessionState.negotiatedSilenceMs = 600;
resetRealtimeTelemetry(sessionState);
assert(sessionState.bargeInCount === 0 &&
  sessionState.confirmedEchoSuppression === null &&
  sessionState.negotiatedSilenceMs === 400,
  'disconnect/reconnect reset must clear barge-in count and confirmed echo state');
assert(formatRealtimeBargeInLabel(-1) === 'Barge-ins: 0',
  'invalid negative barge-in count must clamp to 0');

console.log('realtime echo-suppression behavior passed');
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "realtime-echo-suppression.mjs"
        path.write_text(harness, encoding="utf-8")
        result = subprocess.run([node, str(path)], capture_output=True,
                                text=True, timeout=60)
    require(result.returncode == 0,
            f"realtime echo-suppression behavior failed: {(result.stderr or result.stdout).strip()[:1200]}")
    require("realtime echo-suppression behavior passed" in result.stdout,
            "realtime echo-suppression harness did not reach its success assertion")


def check_showcase_and_telemetry(html, script):
    """North-Star showcase starter presets and live per-turn telemetry hooks."""
    for snippet in (
        'id="emptyState"',
        'id="showcasePresets"',
        'data-preset="documentary"',
        'Live Documentary',
        'Interleaved prose, diagram &amp; voice',
        'data-preset="storyboard_i2v"',
        'Storyboard I2V Chain',
        'Slot 0 keyframe -&gt; Slot 1 anchored I2V',
        'data-preset="smart_text_only"',
        'Smart Text-Only',
        'Autonomous zero-visual abstention',
        '.telemetry-strip',
        '.telemetry-pill',
    ):
        require(snippet in html, f"missing showcase/telemetry markup or CSS: {snippet}")
    for hook in (
        'SHOWCASE_PRESETS',
        'applyShowcasePresetToControls(',
        'updateEmptyStateVisibility(',
        'formatTurnTelemetry(',
        'renderTurnTelemetry(',
        'recordTurnTotal(',
        'recordVisualSettled(',
        'recordFirstAudio(',
        'performance.now()',
        'telemetry-strip',
        'telemetry-pill',
    ):
        require(hook in script, f"missing showcase/telemetry script hook: {hook}")


def check_showcase_and_telemetry_behavior(script):
    """Execute showcase presets and turn telemetry formatters in Node."""
    node = shutil.which("node")
    if not node:
        print("skip showcase/telemetry behavior: node is not installed", flush=True)
        return
    require("// SHOWCASE_PRESETS_BEGIN" in script
            and "// SHOWCASE_PRESETS_END" in script,
            "missing showcase presets behavior-test markers")
    require("// TURN_TELEMETRY_BEGIN" in script
            and "// TURN_TELEMETRY_END" in script,
            "missing turn telemetry behavior-test markers")
    presets_src = script.split("// SHOWCASE_PRESETS_BEGIN", 1)[1].split(
        "// SHOWCASE_PRESETS_END", 1)[0]
    telemetry_src = script.split("// TURN_TELEMETRY_BEGIN", 1)[1].split(
        "// TURN_TELEMETRY_END", 1)[0]
    harness = presets_src + "\n" + telemetry_src + r"""
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

// 1. Showcase preset application across all 3 curated presets.
let focusCount = 0;
let syncCount = 0;
let saveCount = 0;
let audioState = false;
const controls = {
  narrative: { checked: false },
  visualMode: { value: 'image' },
  allowTextOnly: { checked: false },
  narrativeContinuity: { value: 'prompt_anchor' },
  styleAnchor: { value: 'stale' },
  fastVideo: { checked: false },
  prompt: { value: '', focus() { focusCount += 1; } },
};
const hooks = {
  setAudioEnabled(v) { audioState = v; },
  syncNarrativeControls() { syncCount += 1; },
  saveSettings() { saveCount += 1; },
};

const doc = applyShowcasePresetToControls('documentary', controls, hooks);
assert(doc && doc.title === 'Live Documentary' &&
       doc.subtitle === 'Interleaved prose, diagram & voice',
       'documentary metadata mismatch');
assert(controls.narrative.checked === true &&
       controls.visualMode.value === 'auto' &&
       controls.allowTextOnly.checked === true &&
       controls.narrativeContinuity.value === 'off' &&
       controls.styleAnchor.value === '' &&
       controls.fastVideo.checked === true &&
       audioState === true &&
       controls.prompt.value === 'Explain how a black hole accretion disk produces relativistic jets and gravitational lensing. Speak clearly like a live documentary narrator, include one crisp diagram of the spacetime curvature, and one short motion clip of plasma spiraling into the event horizon.',
       'documentary preset control mutations mismatch');
assert(focusCount === 1 && syncCount === 1 && saveCount === 1,
       'documentary preset did not focus/sync/save');

const story = applyShowcasePresetToControls('storyboard_i2v', controls, hooks);
assert(story && story.title === 'Storyboard I2V Chain' &&
       story.subtitle === 'Slot 0 keyframe -> Slot 1 anchored I2V',
       'storyboard_i2v metadata mismatch');
assert(controls.narrative.checked === true &&
       controls.visualMode.value === 'auto' &&
       controls.allowTextOnly.checked === false &&
       controls.narrativeContinuity.value === 'i2v_chain' &&
       controls.styleAnchor.value === 'cinematic 35mm film grain, warm lantern twilight, smooth camera motion' &&
       controls.fastVideo.checked === true &&
       audioState === true &&
       controls.prompt.value === 'Create a two-beat cinematic storyboard of a glowing paper lantern drifting over a misty mountain river at dusk: first illustrate the establishing keyframe, then animate the river breeze and water reflections in motion.',
       'storyboard_i2v preset control mutations mismatch');
assert(focusCount === 2, 'storyboard_i2v preset did not focus prompt');

const smart = applyShowcasePresetToControls('smart_text_only', controls, hooks);
assert(smart && smart.title === 'Smart Text-Only' &&
       smart.subtitle === 'Autonomous zero-visual abstention',
       'smart_text_only metadata mismatch');
assert(controls.narrative.checked === true &&
       controls.visualMode.value === 'auto' &&
       controls.allowTextOnly.checked === true &&
       controls.narrativeContinuity.value === 'off' &&
       controls.styleAnchor.value === '' &&
       controls.fastVideo.checked === false &&
       audioState === false &&
       controls.prompt.value === 'Prove concisely why the square root of 2 is irrational and explain the key step in two sentences.',
       'smart_text_only preset control mutations mismatch');
assert(focusCount === 3, 'smart_text_only preset did not focus prompt');

// Unknown preset is a safe no-op.
assert(applyShowcasePresetToControls('unknown', controls, hooks) === null,
       'unknown preset should return null');

// Empty state visibility toggles cleanly with message count.
const emptyEl = { hidden: false };
updateEmptyStateVisibility(emptyEl, 0);
assert(emptyEl.hidden === false, 'empty state must be visible when messageCount is 0');
updateEmptyStateVisibility(emptyEl, 2);
assert(emptyEl.hidden === true, 'empty state must hide when messageCount > 0');
updateEmptyStateVisibility(emptyEl, 0);
assert(emptyEl.hidden === false, 'empty state must reappear after New chat');

// 2. formatTurnTelemetry pure formatting and zero-fabrication guarantees.
assert(formatTurnTelemetry(null).length === 0, 'null telemetry must yield empty pills');
assert(formatTurnTelemetry({}).length === 0, 'empty telemetry must yield empty pills');
assert(formatTurnTelemetry({
  ttftMs: null, ttfaMs: undefined, visualSettledCount: 0,
  firstVisualMs: null, hasI2VChain: false, totalMs: null,
}).length === 0, 'unrecorded telemetry must yield zero pills');

// Invalid / negative / non-finite values must be omitted, never fabricated.
assert(formatTurnTelemetry({
  ttftMs: -5, ttfaMs: Number.NaN, visualSettledCount: 0,
  firstVisualMs: Number.POSITIVE_INFINITY, hasI2VChain: false, totalMs: -1,
}).length === 0, 'invalid numeric timings must be omitted');

// Full multimodal I2V turn: sub-second ms and >= 1s formatting.
const fullPills = formatTurnTelemetry({
  ttftMs: 265,
  ttfaMs: 380,
  visualSettledCount: 2,
  firstVisualMs: 4120,
  hasI2VChain: true,
  totalMs: 8790,
});
assert(JSON.stringify(fullPills.map((p) => p.label)) ===
  '["TTFT 265 ms","TTFA 380 ms","2 visuals (4.12 s)","I2V","Total 8.79 s"]',
  'full telemetry pill formatting mismatch: ' + JSON.stringify(fullPills));

// Explicit anchor upgrades the I2V pill cleanly while keeping metric kind 'i2v'.
const explicitPills = formatTurnTelemetry({
  ttftMs: 210,
  visualSettledCount: 2,
  firstVisualMs: 3200,
  hasI2VChain: true,
  hasExplicitAnchor: true,
  totalMs: 9100,
});
assert(JSON.stringify(explicitPills.map((p) => p.label)) ===
  '["TTFT 210 ms","2 visuals (3.20 s)","Explicit I2V","Total 9.10 s"]',
  'explicit anchor telemetry pill formatting mismatch: ' + JSON.stringify(explicitPills));
assert(explicitPills.find((p) => p.label === 'Explicit I2V')?.kind === 'i2v',
  'Explicit I2V pill must keep kind i2v');

// Text-only turn omits TTFA, visuals, and I2V.
const textOnlyPills = formatTurnTelemetry({
  ttftMs: 999.6,
  ttfaMs: null,
  visualSettledCount: 0,
  firstVisualMs: null,
  hasI2VChain: false,
  totalMs: 1420,
});
assert(JSON.stringify(textOnlyPills.map((p) => p.label)) ===
  '["TTFT 1.00 s","Total 1.42 s"]',
  'text-only / boundary telemetry mismatch: ' + JSON.stringify(textOnlyPills));

// Singular visual without total (still streaming) and sub-second 0 ms bound.
const streamingVisualPills = formatTurnTelemetry({
  ttftMs: 0,
  visualSettledCount: 1,
  firstVisualMs: 850,
  hasI2VChain: false,
  totalMs: null,
});
assert(JSON.stringify(streamingVisualPills.map((p) => p.label)) ===
  '["TTFT 0 ms","1 visual (850 ms)"]',
  'streaming visual telemetry mismatch: ' + JSON.stringify(streamingVisualPills));

console.log('showcase and telemetry behavior passed');
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "showcase-telemetry.mjs"
        path.write_text(harness, encoding="utf-8")
        result = subprocess.run([node, str(path)], capture_output=True,
                                text=True, timeout=60)
    require(result.returncode == 0,
            f"showcase/telemetry behavior failed: {(result.stderr or result.stdout).strip()[:1200]}")
    require("showcase and telemetry behavior passed" in result.stdout,
            "showcase/telemetry harness did not reach its success assertion")


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


class WebUIContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = read_page()
        scripts = module_scripts(cls.html)
        require(len(scripts) == 1, f"expected exactly one module script, found {len(scripts)}")
        cls.script = scripts[0]

    def test_markup(self):
        check_markup(self.html)

    def test_syntax(self):
        check_node(self.script)

    def test_hooks(self):
        check_hooks(self.script)

    def test_wave10(self):
        check_wave10(self.script)

    def test_wave11(self):
        check_wave11(self.script)

    def test_markdown(self):
        check_markdown(self.html, self.script)

    def test_video_input(self):
        check_video_input(self.html, self.script)

    def test_video_frame_selection_behavior(self):
        check_video_frame_selection_behavior(self.script)

    def test_narrative_continuity(self):
        check_narrative_continuity(self.html, self.script)

    def test_narrative_continuity_behavior(self):
        check_narrative_continuity_behavior(self.script)

    def test_realtime(self):
        check_realtime(self.html, self.script)

    def test_realtime_behavior(self):
        check_realtime_behavior(self.script)

    def test_realtime_echo_suppression_behavior(self):
        check_realtime_echo_suppression_behavior(self.script)

    def test_showcase_and_telemetry(self):
        check_showcase_and_telemetry(self.html, self.script)

    def test_showcase_and_telemetry_behavior(self):
        check_showcase_and_telemetry_behavior(self.script)

    def test_theme(self):
        check_theme_tokens(self.html)


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
        ("video_input", lambda: check_video_input(html, script)),
        ("video_frame_selection_behavior", lambda: check_video_frame_selection_behavior(script)),
        ("narrative_continuity", lambda: check_narrative_continuity(html, script)),
        ("narrative_continuity_behavior", lambda: check_narrative_continuity_behavior(script)),
        ("realtime", lambda: check_realtime(html, script)),
        ("realtime_behavior", lambda: check_realtime_behavior(script)),
        ("realtime_echo_suppression_behavior", lambda: check_realtime_echo_suppression_behavior(script)),
        ("showcase_and_telemetry", lambda: check_showcase_and_telemetry(html, script)),
        ("showcase_and_telemetry_behavior", lambda: check_showcase_and_telemetry_behavior(script)),
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

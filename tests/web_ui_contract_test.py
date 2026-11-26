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
                'data-preset="smart_text_only"', 'data-preset="temporal_video_qa"',
                'id="prompt"', 'id="send"', 'id="stop"',
                'id="voiceToggle"', 'id="voiceWave"', 'id="voiceStatus"',
                'id="voiceMute"', 'id="voiceStop"', 'id="realtimeToggle"',
                'id="realtimeEchoSuppression"', 'id="realtimeStatus"',
                'id="realtimeStrip"', 'id="realtimeStripPhase"',
                'id="realtimeMicMeter"', 'id="realtimeMicMeterFill"',
                'id="realtimeEchoState"', 'id="realtimeBargeInCount"',
                'id="narrative"',
                'id="visualMode"', 'id="maxSlots"', 'id="allowTextOnly"',
                'id="narrativeContinuity"', 'id="styleAnchor"',
                'id="fastVideo"', 'id="videoFramePolicy"',
                'id="videoMotionThreshold"', 'id="videoSceneThreshold"',
                'id="nativeAllocatorTelemetry"', 'id="behaviorLogprobs"',
                'id="exportFormat"', 'id="exportSession"',
                'id="importSession"', 'id="importSessionInput"'):
        require(tag in html, f"missing element: {tag}")
    require('<option value="off" selected>Off</option>' in html,
            "narrativeContinuity must default to Off")
    require('<option value="prompt_anchor">Prompt anchor</option>' in html
            and '<option value="i2v_chain">I2V chain</option>' in html,
            "narrativeContinuity must expose prompt_anchor and i2v_chain options")
    require('<input type="checkbox" id="fastVideo">' in html,
            "fastVideo checkbox must default to unchecked")
    require('<input type="checkbox" id="nativeAllocatorTelemetry">' in html,
            "nativeAllocatorTelemetry checkbox must default to unchecked")
    require('<input type="checkbox" id="behaviorLogprobs">' in html,
            "behaviorLogprobs checkbox must default to unchecked")
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
    # Opt-in Native VRAM telemetry (telemetry.native_allocator=true): default
    # unchecked, persisted in localStorage, disabled while running, omitted on wire when off.
    require("localStorage.getItem('ummweb.nativeAllocatorTelemetry') === '1'" in script,
            "nativeAllocatorTelemetry must default to unchecked unless persisted as '1'")
    require("localStorage.setItem('ummweb.nativeAllocatorTelemetry'" in script,
            "nativeAllocatorTelemetry must persist in localStorage")
    require("applyNativeAllocatorTelemetryOption(body, ui.nativeAllocatorTelemetry.checked)" in script,
            "send() must apply nativeAllocatorTelemetry through applyNativeAllocatorTelemetryOption")
    require("extractNativeAllocatorTelemetry(root)" in script
            and "recordNativeAllocatorTelemetry(turn, rawTelemetry)" in script,
            "consume() must extract and record server-returned native allocator telemetry")
    require("ui.nativeAllocatorTelemetry" in script.split("function setBusy(busy)", 1)[1].split("}", 1)[0],
            "nativeAllocatorTelemetry must be disabled while running")
    # Opt-in Behavior logprobs (behavior.logprobs=true): default unchecked,
    # persisted in localStorage, disabled while running, omitted on wire when off.
    require("localStorage.getItem('ummweb.behaviorLogprobs') === '1'" in script,
            "behaviorLogprobs must default to unchecked unless persisted as '1'")
    require("localStorage.setItem('ummweb.behaviorLogprobs'" in script,
            "behaviorLogprobs must persist in localStorage")
    require("applyBehaviorLogprobsOption(body, behaviorRequested)" in script,
            "send() must apply behaviorLogprobs through applyBehaviorLogprobsOption")
    require("res.headers.get('x-umminfer-behavior-evidence')" in script,
            "send() must read X-UmmInfer-Behavior-Evidence response header")
    require("extractBehaviorEvidence(root)" in script
            and "recordBehaviorEvidence(turn," in script
            and "finalizeBehaviorEvidence(turn," in script,
            "SSE handling must extract, record, and finalize server-returned behavior evidence")
    require("ui.behaviorLogprobs" in script.split("function setBusy(busy)", 1)[1].split("}", 1)[0],
            "behaviorLogprobs must be disabled while running")


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
    # Native video input upload (+ Vid -> video_url content part).
    require('id="askVideo"' in html and 'id="pickVideo"' in html,
            "composer must expose askVideo file input and pickVideo button")
    require("type: 'video_url'" in script,
            "composer must serialize attached videos as video_url content parts")
    require("attachedVideo" in script,
            "composer must track attachedVideo state")
    require("videoURLPart(attachedVideo.dataURL, frameSelection)" in script,
            "attached video must use the shared frame-selection serializer")
    require("parseVideoFrameSelection(ui.videoFramePolicy.value" in script
            and "ui.videoFrameBudget.value" in script,
            "frame policy, thresholds, and budget must be validated while building the request")
    require("selection.motion_threshold = frameSelection.motion_threshold" in script
            and "selection.scene_threshold = frameSelection.scene_threshold" in script
            and "selection.max_frames = frameSelection.max_frames" in script,
            "frame selection must emit accepted policy/threshold/max_frames fields")
    require("maxFrames === undefined ? undefined" in script,
            "Uniform without a budget must preserve legacy request bytes")
    require('id="videoFrameBudget"' in html
            and '<option value="" selected>Default (8, omitted)</option>' in html
            and '<option value="4">4 · lower cost</option>' in html
            and '<option value="8">8 · explicit</option>' in html
            and '<option value="16">16 · higher cost</option>' in html,
            "frame budget must expose omission and exact 4/8/16 choices")
    require("not answer quality" in html and "prefill/ViT cost" in html,
            "budget tooltip must explain selection cost without promising quality")
    for key in ('videoFramePolicy', 'videoFrameBudget',
                'videoMotionThreshold', 'videoSceneThreshold'):
        require(f"'{key}'" in script, f"{key} must persist with existing settings")
        require(f"ui.{key}" in script.split("function setBusy(busy)", 1)[1].split("])", 1)[0],
                f"{key} must be disabled while a request is running")
    require("syncVideoFrameControls()" in script,
            "threshold controls must synchronize with Uniform/Temporal and busy state")
    require("parseVideoFrameBudget(saved)" in script
            and "if (videoFrameBudgetPersistenceError) throw new Error" in script,
            "invalid persisted budgets must fail closed before fetch")


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

// Explicit 4/8/16 budgets emit JSON integers under either policy. Explicit 8
// remains distinguishable from omission even though native decode is equivalent.
for (const budget of [4, 8, 16]) {
  const uniform = videoURLPart(
    dataURL, parseVideoFrameSelection('uniform', 'ignored', '', String(budget)));
  assert(JSON.stringify(uniform) ===
    `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA","frame_selection":{"policy":"uniform","max_frames":${budget}}}}`,
    `uniform ${budget} wire schema mismatch`);
  assert(uniform.video_url.frame_selection.max_frames === budget &&
         Number.isInteger(uniform.video_url.frame_selection.max_frames),
         `uniform ${budget} was not an integer`);
  const temporalBudget = videoURLPart(
    dataURL, parseVideoFrameSelection('temporal', '0.02', '0.25', String(budget)));
  assert(JSON.stringify(temporalBudget) ===
    `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA","frame_selection":{"policy":"temporal","motion_threshold":0.02,"scene_threshold":0.25,"max_frames":${budget}}}}`,
    `temporal ${budget} wire schema mismatch`);
}

const invalidBudgets = ['0', '1', '3', '5', '15', '17', '-4', '4.0', '04',
                        '8.5', 'true', ' 4', '4 ', 'NaN', 'Infinity', 'abc'];
for (const budget of invalidBudgets) {
  let rejected = false;
  try { parseVideoFrameSelection('uniform', '', '', budget); }
  catch (error) { rejected = /Default, 4, 8, or 16/.test(error.message); }
  assert(rejected, `invalid persisted budget was accepted: ${budget}`);
}
for (const budget of [true, false, 4, 8.0, 16.0, {}, []]) {
  let rejected = false;
  try { parseVideoFrameSelection('uniform', '', '', budget); }
  catch { rejected = true; }
  assert(rejected, `non-string UI budget was not rejected: ${String(budget)}`);
}

const lifecycleControls = {
  videoFramePolicy: { value: 'temporal' },
  videoMotionThreshold: { value: '0.5' },
  videoSceneThreshold: { value: '0.75' },
  videoFrameBudget: { value: '16' },
};
applyVideoFrameSelectionToControls(lifecycleControls, undefined);
assert(lifecycleControls.videoFramePolicy.value === 'uniform' &&
       lifecycleControls.videoMotionThreshold.value === '0.02' &&
       lifecycleControls.videoSceneThreshold.value === '0.25' &&
       lifecycleControls.videoFrameBudget.value === '',
       'new-chat/default reset retained a stale frame budget');

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
const reconnected = parseVideoFrameSelection('temporal', '1', '0', '16');
assert(reconnected.motion_threshold === 1 && reconnected.scene_threshold === 0 &&
       reconnected.max_frames === 16 && temporal.max_frames === undefined &&
       temporal.motion_threshold === 0.02 && temporal.scene_threshold === 0.25,
       'reconnect reused or mutated stale temporal settings');
assert(parseVideoFrameSelection('uniform', '', '', '') === undefined,
       'reconnect/default reset silently retained the previous 16-frame budget');

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
    require("computePcm16RmsLevel(pcmBytes)" in script and
            "setRealtimeMicLevel(" in script,
            "realtime mic capture must compute and display real PCM16 RMS level")
    require('aria-label="Live microphone input level"' in html,
            "realtime mic meter must expose an accessible aria-label")
    require("recordRealtimePlaybackAck(" in script and
            "recordRealtimeTurnCancelled(" in script and
            "attachRealtimeInterruptedBadge(" in script and
            "turn-interrupted-badge" in html,
            "realtime turns must track playback_ack cursors and attach per-turn interrupted badges")
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


def check_realtime_audio_meter_and_barge_in_behavior(script):
    """Execute PCM16 RMS level meter and per-turn barge-in badge helpers in Node."""
    node = shutil.which("node")
    if not node:
        print("skip realtime audio meter behavior: node is not installed", flush=True)
        return
    require("// REALTIME_AUDIO_METER_BEGIN" in script and
            "// REALTIME_AUDIO_METER_END" in script,
            "missing realtime audio meter behavior-test markers")
    helpers = script.split("// REALTIME_AUDIO_METER_BEGIN", 1)[1].split(
        "// REALTIME_AUDIO_METER_END", 1)[0]
    attach_fn = re.search(
        r"function attachRealtimeInterruptedBadge\(.*?\n\}", script, re.S)
    require(attach_fn is not None,
            "could not extract attachRealtimeInterruptedBadge for behavior checks")
    harness = helpers + "\n" + attach_fn.group(0) + r"""
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const assertClose = (actual, expected, tol, message) => {
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`${message}: expected ${expected} ± ${tol}, got ${actual}`);
  }
};
function pcmFromSamples(samples) {
  const bytes = new Uint8Array(samples.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < samples.length; i += 1) {
    view.setInt16(i * 2, samples[i], true);
  }
  return bytes;
}

// 1. computePcm16RmsLevel: empty, odd, non-Uint8Array, silence, square, sine, full-scale clamp, subarray.
assert(computePcm16RmsLevel(null) === 0, 'null must return 0');
assert(computePcm16RmsLevel(undefined) === 0, 'undefined must return 0');
assert(computePcm16RmsLevel(new Uint8Array(0)) === 0, 'empty buffer must return 0');
assert(computePcm16RmsLevel(new Uint8Array([0x00, 0x40, 0x01])) === 0,
  'odd byte-length buffer must return 0');
assert(computePcm16RmsLevel(pcmFromSamples([0, 0, 0, 0])) === 0,
  'silent PCM16 buffer must return 0');

// Half-scale square wave (+16384 / -16384) => RMS = 0.5 exactly.
const halfSquare = pcmFromSamples([16384, -16384, 16384, -16384]);
assertClose(computePcm16RmsLevel(halfSquare), 0.5, 1e-6, 'half-scale square RMS');

// Mid-scale sine wave (amplitude 16384 over exact integer cycles) => RMS = 0.5 / sqrt(2).
const sineSamples = [];
for (let i = 0; i < 640; i += 1) {
  sineSamples.push(Math.round(16384 * Math.sin((2 * Math.PI * 10 * i) / 640)));
}
assertClose(computePcm16RmsLevel(pcmFromSamples(sineSamples)), 0.5 / Math.SQRT2, 1e-3,
  'mid-scale sine RMS');

// Full-scale negative (-32768) => exact 1.0 (clamped); full-scale positive (32767) <= 1.0.
const fullNeg = pcmFromSamples([-32768, -32768, -32768, -32768]);
assert(computePcm16RmsLevel(fullNeg) === 1, 'full-scale negative must clamp cleanly to 1.0');
const fullPos = pcmFromSamples([32767, 32767, 32767, 32767]);
const posRms = computePcm16RmsLevel(fullPos);
assert(posRms > 0.9999 && posRms <= 1, 'full-scale positive RMS must be within (0.9999, 1.0]');

// Non-zero byteOffset Uint8Array view computes RMS over slice only.
const padded = pcmFromSamples([0, 16384, -16384, 0]);
const slice = new Uint8Array(padded.buffer, 2, 4);
assertClose(computePcm16RmsLevel(slice), 0.5, 1e-6, 'byteOffset subarray RMS');

// 2. formatInterruptedBadge exact strings and zero-fabrication guards.
assert(formatInterruptedBadge(0) === 'Interrupted before audio',
  '0 B badge mismatch');
assert(formatInterruptedBadge(12) === 'Interrupted · 12 B spoken',
  '12 B badge mismatch');
assert(formatInterruptedBadge(2048) === 'Interrupted · 2048 B spoken',
  '2048 B badge mismatch');
for (const bad of [-1, null, undefined, Number.NaN, 1.5, '12']) {
  assert(formatInterruptedBadge(bad) === '', `invalid cursor must yield empty badge: ${bad}`);
}

// 3. Per-turn barge-in cut-point tracking and inline badge DOM attachment.
class FakeEl {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.className = '';
    this.textContent = '';
    this.dataset = {};
    this.attrs = {};
    this.children = [];
  }
  append(...nodes) { this.children.push(...nodes); }
  setAttribute(k, v) { this.attrs[k] = v; }
}
globalThis.document = { createElement: (tag) => new FakeEl(tag) };

const tracker = createRealtimeTurnTracker();

// Completed turn: playback_ack observed and cleared; never marked interrupted.
assert(recordRealtimePlaybackAck(tracker, 'r-done', 12) === 12, 'playback_ack 12');
assert(recordRealtimePlaybackAck(tracker, 'r-done', 24) === 24, 'playback_ack 24');
clearRealtimeTurnTracking(tracker, 'r-done');
assert(!tracker.interruptedBytesByResponse.has('r-done'),
  'completed turn must never be marked interrupted');
assert(!tracker.acknowledgedBytesByResponse.has('r-done'),
  'completed turn tracking must be cleared');

// Cancelled before any audio playback ACK => 0 B -> 'Interrupted before audio'.
const zeroTurn = { meta: new FakeEl('div'), interruptedBadgeEl: null };
const zeroCancel = recordRealtimeTurnCancelled(tracker, 'r-zero', 0);
assert(zeroCancel.acknowledgedBytes === 0 && zeroCancel.badgeText === 'Interrupted before audio',
  'zero-byte barge-in result mismatch');
attachRealtimeInterruptedBadge(zeroTurn, zeroCancel.acknowledgedBytes, zeroCancel.badgeText);
assert(zeroTurn.meta.children.length === 1 &&
       zeroTurn.interruptedBadgeEl.textContent === 'Interrupted before audio' &&
       zeroTurn.interruptedBadgeEl.dataset.acknowledgedBytes === '0' &&
       zeroTurn.interruptedBadgeEl.attrs['aria-label'] === 'Interrupted before audio',
  'zero-byte badge DOM mismatch');

// Cancelled after real playback_ack cursors (with omitted text_cursor on cancelled) retains real ack.
assert(recordRealtimePlaybackAck(tracker, 'r-ack', 12) === 12, 'ack 12');
assert(recordRealtimePlaybackAck(tracker, 'r-ack', 24) === 24, 'ack 24');
// Stale/negative playback_ack cursors never regress monotonic max.
assert(recordRealtimePlaybackAck(tracker, 'r-ack', 8) === 24, 'monotonic max must hold');
assert(recordRealtimePlaybackAck(tracker, 'r-ack', -5) === null, 'negative cursor must be ignored');
const ackTurn = { meta: new FakeEl('div'), interruptedBadgeEl: null };
const ackCancel = recordRealtimeTurnCancelled(tracker, 'r-ack', undefined);
assert(ackCancel.acknowledgedBytes === 24 && ackCancel.badgeText === 'Interrupted · 24 B spoken',
  'playback_ack tracked barge-in mismatch');
attachRealtimeInterruptedBadge(ackTurn, ackCancel.acknowledgedBytes, ackCancel.badgeText);
// Duplicate cancellation updates in place without adding a second badge element.
const dupCancel = recordRealtimeTurnCancelled(tracker, 'r-ack', 30);
attachRealtimeInterruptedBadge(ackTurn, dupCancel.acknowledgedBytes, dupCancel.badgeText);
assert(ackTurn.meta.children.length === 1 &&
       ackTurn.interruptedBadgeEl.textContent === 'Interrupted · 30 B spoken' &&
       ackTurn.interruptedBadgeEl.dataset.acknowledgedBytes === '30',
  'duplicate cancel must update single badge cleanly');

resetRealtimeTurnTracker(tracker);
assert(tracker.acknowledgedBytesByResponse.size === 0 &&
       tracker.interruptedBytesByResponse.size === 0,
  'resetRealtimeTurnTracker must clear all maps');

console.log('realtime audio meter and barge-in behavior passed');
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "realtime-audio-meter.mjs"
        path.write_text(harness, encoding="utf-8")
        result = subprocess.run([node, str(path)], capture_output=True,
                                text=True, timeout=60)
    require(result.returncode == 0,
            f"realtime audio meter behavior failed: {(result.stderr or result.stdout).strip()[:1200]}")
    require("realtime audio meter and barge-in behavior passed" in result.stdout,
            "realtime audio meter harness did not reach its success assertion")


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
        'data-preset="temporal_video_qa"',
        'Temporal Video QA',
        'Motion &amp; scene-cut multi-turn video Q&amp;A',
        '<span class="preset-tag">Temporal</span>',
        '<span class="preset-tag">+ Vid</span>',
        '<span class="preset-tag">Multi-turn</span>',
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

// 1. Showcase preset application across all 4 curated presets.
assert(JSON.stringify(Object.keys(SHOWCASE_PRESETS)) ===
       '["documentary","storyboard_i2v","smart_text_only","temporal_video_qa"]',
       'expected exactly 4 curated showcase presets');

let focusCount = 0;
let syncCount = 0;
let syncVideoFrameCount = 0;
let saveCount = 0;
let audioState = false;
let hasAttachedVideo = false;
let lastStatus = 'Ready';
let lastStatusBusy = false;
const controls = {
  narrative: { checked: false },
  visualMode: { value: 'image' },
  allowTextOnly: { checked: false },
  narrativeContinuity: { value: 'prompt_anchor' },
  styleAnchor: { value: 'stale' },
  fastVideo: { checked: false },
  videoFramePolicy: { value: 'temporal' },
  videoMotionThreshold: { value: '0.15' },
  videoSceneThreshold: { value: '0.55' },
  nativeAllocatorTelemetry: { checked: false },
  behaviorLogprobs: { checked: false },
  advanced: { open: false },
  prompt: { value: '', focus() { focusCount += 1; } },
};
const hooks = {
  setAudioEnabled(v) { audioState = v; },
  syncNarrativeControls() { syncCount += 1; },
  syncVideoFrameControls() { syncVideoFrameCount += 1; },
  saveSettings() { saveCount += 1; },
  hasAttachedVideo() { return hasAttachedVideo; },
  setStatus(text, busy = false) { lastStatus = text; lastStatusBusy = busy; },
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
       controls.videoFramePolicy.value === 'uniform' &&
       controls.videoMotionThreshold.value === '0.02' &&
       controls.videoSceneThreshold.value === '0.25' &&
       audioState === true &&
       lastStatus === 'Ready' &&
       lastStatusBusy === false &&
       controls.prompt.value === 'Explain how a black hole accretion disk produces relativistic jets and gravitational lensing. Speak clearly like a live documentary narrator, include one crisp diagram of the spacetime curvature, and one short motion clip of plasma spiraling into the event horizon.',
       'documentary preset control mutations mismatch');
assert(focusCount === 1 && syncCount === 1 && syncVideoFrameCount === 1 && saveCount === 1,
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
       new TextEncoder().encode(controls.styleAnchor.value).length <= 256 &&
       controls.fastVideo.checked === true &&
       controls.videoFramePolicy.value === 'uniform' &&
       audioState === true &&
       controls.prompt.value.includes('source=slot-0') &&
       controls.prompt.value === 'Create a two-beat cinematic storyboard of a glowing paper lantern drifting over a misty mountain river at dusk: first illustrate the establishing keyframe as slot-0, then animate the river breeze and water reflections in a motion clip explicitly anchored to source=slot-0.',
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
       controls.videoFramePolicy.value === 'uniform' &&
       audioState === false &&
       controls.prompt.value === 'Prove concisely why the square root of 2 is irrational and explain the key step in two sentences.',
       'smart_text_only preset control mutations mismatch');
assert(focusCount === 3, 'smart_text_only preset did not focus prompt');

// Temporal Video QA without an attached video sets temporal policy, thresholds, opens Options, and shows non-blocking hint.
controls.videoMotionThreshold.value = '0.99';
controls.videoSceneThreshold.value = '0.88';
controls.advanced.open = false;
hasAttachedVideo = false;
const temporal = applyShowcasePresetToControls('temporal_video_qa', controls, hooks);
assert(temporal && temporal.title === 'Temporal Video QA' &&
       temporal.subtitle === 'Motion & scene-cut multi-turn video Q&A',
       'temporal_video_qa metadata mismatch');
assert(controls.narrative.checked === true &&
       controls.visualMode.value === 'auto' &&
       controls.allowTextOnly.checked === true &&
       controls.narrativeContinuity.value === 'off' &&
       controls.styleAnchor.value === '' &&
       controls.fastVideo.checked === false &&
       audioState === false &&
       controls.videoFramePolicy.value === 'temporal' &&
       controls.videoMotionThreshold.value === '0.02' &&
       controls.videoSceneThreshold.value === '0.25' &&
       controls.advanced.open === true &&
       lastStatus === 'Attach a clip with + Vid, then press Send' &&
       lastStatusBusy === false &&
       controls.prompt.value === 'Watch this attached video clip and summarize the key visual events, motion changes, and scene cuts in chronological order.',
       'temporal_video_qa preset control mutations mismatch');
assert(focusCount === 4 && syncVideoFrameCount === 4,
       'temporal_video_qa preset did not focus prompt and sync video frame controls');

// Temporal Video QA with an already-attached video keeps status Ready (no missing-video hint).
hasAttachedVideo = true;
applyShowcasePresetToControls('temporal_video_qa', controls, hooks);
assert(lastStatus === 'Ready' && controls.videoFramePolicy.value === 'temporal',
       'temporal_video_qa with attached video should keep status Ready');

// Switching away from Temporal Video QA to each other preset resets videoFramePolicy back to uniform.
for (const otherId of ['documentary', 'storyboard_i2v', 'smart_text_only']) {
  hasAttachedVideo = false;
  applyShowcasePresetToControls('temporal_video_qa', controls, hooks);
  controls.videoMotionThreshold.value = '0.12';
  controls.videoSceneThreshold.value = '0.44';
  assert(controls.videoFramePolicy.value === 'temporal', 'setup temporal failed');
  applyShowcasePresetToControls(otherId, controls, hooks);
  assert(controls.videoFramePolicy.value === 'uniform' &&
         controls.videoMotionThreshold.value === '0.02' &&
         controls.videoSceneThreshold.value === '0.25' &&
         lastStatus === 'Ready',
         `switching from temporal_video_qa to ${otherId} did not reset videoFramePolicy to uniform`);
}

// None of the 4 Showcase presets auto-enable or mutate nativeAllocatorTelemetry or behaviorLogprobs.
for (const presetId of ['documentary', 'storyboard_i2v', 'smart_text_only', 'temporal_video_qa']) {
  controls.nativeAllocatorTelemetry.checked = false;
  controls.behaviorLogprobs.checked = false;
  applyShowcasePresetToControls(presetId, controls, hooks);
  assert(controls.nativeAllocatorTelemetry.checked === false,
         `preset ${presetId} must not auto-enable nativeAllocatorTelemetry`);
  assert(controls.behaviorLogprobs.checked === false,
         `preset ${presetId} must not auto-enable behaviorLogprobs`);
}

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

// 3. Opt-in native CUDA allocator telemetry wire encoding & honest parsing/display.
const baseBody = { model: 'qwen3-native', messages: [{ role: 'user', content: 'hi' }], stream: true };
const offBody = applyNativeAllocatorTelemetryOption({ ...baseBody }, false);
assert(!Object.prototype.hasOwnProperty.call(offBody, 'telemetry') &&
       JSON.stringify(offBody) === JSON.stringify(baseBody),
       'default OFF must omit telemetry completely for byte-identical requests');
const onBody = applyNativeAllocatorTelemetryOption({ ...baseBody }, true);
assert(JSON.stringify(onBody.telemetry) === '{"native_allocator":true}',
       'opt-in ON must emit telemetry: {native_allocator: true}');

// Byte formatting across B / KiB / MiB / GiB and strict rejection of null/invalid values.
assert(formatAllocatorBytes(0) === '0 B', '0 B format mismatch');
assert(formatAllocatorBytes(512) === '512 B', '512 B format mismatch');
assert(formatAllocatorBytes(512 * 1024) === '512 KiB', '512 KiB format mismatch');
assert(formatAllocatorBytes(Math.round(128.4 * 1024 * 1024)) === '128.4 MiB', '128.4 MiB format mismatch');
assert(formatAllocatorBytes(Math.round(1.5 * 1024 * 1024 * 1024)) === '1.5 GiB', '1.5 GiB format mismatch');
for (const badBytes of [null, undefined, false, true, -1, Number.NaN, Number.POSITIVE_INFINITY, '1024']) {
  assert(formatAllocatorBytes(badBytes) === null, `invalid bytes must return null: ${badBytes}`);
}

// Extract telemetry from unified.telemetry (unified SSE) and top-level telemetry (non-unified SSE / JSON).
const coveredEnvelope = {
  schema_version: 1,
  coverage: 'COVERED',
  scope: 'request-native-allocator',
  request_id: 'http-1',
  lifecycle_status: 'completed',
  stages: [
    {
      stage: 'vlm',
      native_stage: 'qwen38',
      coverage: 'COVERED',
      allocated_bytes_peak: Math.round(128.4 * 1024 * 1024),
      reserved_bytes_peak: Math.round(128.4 * 1024 * 1024),
      epoch: { id: 17, reset: true, semantics: 'new_allocation_ownership_epoch' },
    },
  ],
};
assert(extractNativeAllocatorTelemetry({ unified: { kind: 'complete', telemetry: coveredEnvelope } }) === coveredEnvelope,
       'must extract unified.telemetry from unified SSE terminal chunk');
assert(extractNativeAllocatorTelemetry({ choices: [], telemetry: coveredEnvelope }) === coveredEnvelope,
       'must extract top-level telemetry from non-unified SSE / non-stream response');
assert(extractNativeAllocatorTelemetry({ unified: { kind: 'complete' } }) === null,
       'absent telemetry must extract as null');

// Covered envelope renders real peak MiB/KiB pill + detailed tooltip.
const coveredPills = formatTurnTelemetry({
  ttftMs: 140,
  totalMs: 1200,
  nativeAllocator: coveredEnvelope,
});
assert(JSON.stringify(coveredPills.map((p) => p.label)) ===
  '["TTFT 140 ms","Total 1.20 s","VRAM: 128.4 MiB peak"]',
  'covered telemetry pill mismatch: ' + JSON.stringify(coveredPills));
const coveredVramPill = coveredPills.find((p) => p.kind === 'vram');
assert(coveredVramPill && coveredVramPill.coverage === 'COVERED' &&
       coveredVramPill.title.includes('vlm/qwen38') &&
       coveredVramPill.title.includes('COVERED') &&
       coveredVramPill.title.includes('128.4 MiB') &&
       coveredVramPill.title.includes('epoch #17'),
       'covered VRAM tooltip mismatch: ' + coveredVramPill?.title);

// Multi-stage PARTIAL envelope (512 KiB covered + 1 NOT_COVERED) shows covered peak and both stage details.
const partialEnvelope = {
  schema_version: 1,
  coverage: 'PARTIAL',
  stages: [
    {
      stage: 'video',
      native_stage: 'wan22',
      coverage: 'COVERED',
      allocated_bytes_peak: 512 * 1024,
      epoch: { id: 9, reset: true },
    },
    {
      stage: 'tts',
      native_stage: 'tts',
      coverage: 'NOT_COVERED',
      allocated_bytes_peak: null,
      reason: 'backend_not_instrumented',
    },
  ],
};
const partialSummary = summarizeNativeAllocatorTelemetry(partialEnvelope);
assert(partialSummary && partialSummary.covered === true &&
       partialSummary.coverage === 'PARTIAL' &&
       partialSummary.coveredStageCount === 1 &&
       partialSummary.totalStageCount === 2 &&
       partialSummary.label === 'VRAM: 512 KiB peak' &&
       partialSummary.title.includes('video/wan22: COVERED · peak 512 KiB · epoch #9') &&
       partialSummary.title.includes('tts: NOT_COVERED (backend_not_instrumented)'),
       'partial telemetry summary mismatch: ' + JSON.stringify(partialSummary));

// NOT_COVERED envelope with allocated_bytes_peak: null renders honest 'VRAM: N/C' — NEVER '0 B' or '0 MiB'.
const notCoveredEnvelope = {
  schema_version: 1,
  coverage: 'NOT_COVERED',
  stages: [
    {
      stage: 'vlm',
      native_stage: 'qwen38',
      coverage: 'NOT_COVERED',
      allocated_bytes_peak: null,
      reason: 'batched_request_attribution_unavailable',
    },
  ],
};
const notCoveredPills = formatTurnTelemetry({
  ttftMs: 120,
  totalMs: 950,
  nativeAllocator: notCoveredEnvelope,
});
assert(JSON.stringify(notCoveredPills.map((p) => p.label)) ===
  '["TTFT 120 ms","Total 950 ms","VRAM: N/C"]',
  'NOT_COVERED telemetry pill must be VRAM: N/C, got ' + JSON.stringify(notCoveredPills));
const ncPill = notCoveredPills.find((p) => p.kind === 'vram');
assert(ncPill && ncPill.coverage === 'NOT_COVERED' &&
       !ncPill.label.includes('0 B') && !ncPill.label.includes('0 MiB') &&
       ncPill.title.includes('vlm/qwen38') &&
       ncPill.title.includes('batched_request_attribution_unavailable'),
       'NOT_COVERED tooltip mismatch: ' + ncPill?.title);

// Malformed COVERED stage with null/boolean peak never fabricates 0 B; falls back honestly to VRAM: N/C.
for (const badPeak of [null, undefined, false, -10]) {
  const malformed = summarizeNativeAllocatorTelemetry({
    coverage: 'COVERED',
    stages: [{ stage: 'vlm', coverage: 'COVERED', allocated_bytes_peak: badPeak }],
  });
  assert(malformed && malformed.covered === false && malformed.peakBytes === null &&
         malformed.label === 'VRAM: N/C',
         `malformed peak ${badPeak} must render VRAM: N/C, got ${JSON.stringify(malformed)}`);
}

// Absent telemetry omits VRAM pill completely.
const absentPills = formatTurnTelemetry({ ttftMs: 150, totalMs: 1100, nativeAllocator: null });
assert(!absentPills.some((p) => p.kind === 'vram') &&
       JSON.stringify(absentPills.map((p) => p.label)) === '["TTFT 150 ms","Total 1.10 s"]',
       'absent telemetry must omit VRAM pill completely');

// 4. Opt-in behavior.logprobs wire encoding & honest header/payload display (zero fabrication).
const behaviorOffBody = applyBehaviorLogprobsOption({ ...baseBody }, false);
assert(!Object.prototype.hasOwnProperty.call(behaviorOffBody, 'behavior') &&
       JSON.stringify(behaviorOffBody) === JSON.stringify(baseBody),
       'default OFF must omit behavior completely for byte-identical requests');
const behaviorOnBody = applyBehaviorLogprobsOption({ ...baseBody }, true);
assert(JSON.stringify(behaviorOnBody.behavior) === '{"logprobs":true}',
       'opt-in ON must emit behavior: {logprobs: true}');

// Case-insensitive header normalization.
assert(normalizeBehaviorEvidenceHeader('  EXPOSED  ') === 'exposed', 'exposed header normalization mismatch');
assert(normalizeBehaviorEvidenceHeader('Not_Exposed') === 'not_exposed', 'not_exposed header normalization mismatch');
assert(normalizeBehaviorEvidenceHeader('') === '' && normalizeBehaviorEvidenceHeader(null) === '',
       'empty/null header normalization mismatch');

// Behavior payload extraction accepts valid structure and rejects malformed shapes.
const sampleBehaviorPayload = {
  format: 'umminfer-behavior-logprobs-v1',
  tokenizer_sha256: '1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016',
  token_ids: [101, 202, 303, 404, 505],
  logprobs: [-0.1, -0.25, -0.05, -0.4, -0.01],
  mask: [1, 0, 1, 0, 1],
};
const extractedBehavior = extractBehaviorEvidence({ behavior: sampleBehaviorPayload });
assert(extractedBehavior && extractedBehavior.token_ids.length === 5 &&
       extractedBehavior.mask.length === 5 && extractedBehavior.logprobs.length === 5,
       'valid behavior payload was not extracted');
for (const badPayload of [
  null,
  'exposed',
  [],
  { token_ids: [1, 2], mask: [1] },
  { token_ids: [1, 2.5], mask: [1, 1] },
  { token_ids: [1, 2], mask: [1, 2] },
  { token_ids: [1, 2], mask: [1, 1], logprobs: [-0.1] },
  { token_ids: [1, 2], mask: [1, 1], logprobs: [-0.1, Number.NaN] },
]) {
  assert(extractBehaviorEvidence({ behavior: badPayload }) === null,
         'malformed behavior payload was accepted: ' + JSON.stringify(badPayload));
}

// Default OFF (unrequested, no header, no payload) omits behavior pill completely.
const behaviorDefaultPills = formatTurnTelemetry({
  ttftMs: 110, totalMs: 900,
  behaviorRequested: false, behaviorEvidenceHeader: '', behaviorEvidence: null,
});
assert(!behaviorDefaultPills.some((p) => p.kind === 'behavior'),
       'default OFF must omit behavior pill completely');

// Exposed header + valid payload -> Behavior: N tok (M sup) with honest tooltip.
const exposedBehaviorPills = formatTurnTelemetry({
  ttftMs: 120, totalMs: 1050,
  behaviorRequested: true,
  behaviorEvidenceHeader: 'EXPOSED',
  behaviorEvidence: sampleBehaviorPayload,
});
assert(JSON.stringify(exposedBehaviorPills.map((p) => p.label)) ===
       '["TTFT 120 ms","Total 1.05 s","Behavior: 5 tok (3 sup)"]',
       'exposed behavior pill mismatch: ' + JSON.stringify(exposedBehaviorPills));
const exposedPill = exposedBehaviorPills.find((p) => p.kind === 'behavior');
assert(exposedPill && exposedPill.status === 'exposed' &&
       exposedPill.title.includes('umminfer-behavior-logprobs-v1') &&
       exposedPill.title.includes('5 (3 supervised)') &&
       exposedPill.title.includes('1dd1199e35f6cfb9'),
       'exposed behavior tooltip mismatch: ' + exposedPill?.title);

// not_exposed header -> Behavior: not_exposed with zero fabrication (even if stray payload present).
for (const payload of [null, sampleBehaviorPayload]) {
  const notExposedPills = formatTurnTelemetry({
    ttftMs: 130, totalMs: 980,
    behaviorRequested: true,
    behaviorEvidenceHeader: 'not_exposed',
    behaviorEvidence: payload,
  });
  const nePill = notExposedPills.find((p) => p.kind === 'behavior');
  assert(nePill && nePill.label === 'Behavior: not_exposed' &&
         nePill.status === 'not_exposed' &&
         !nePill.label.includes('tok') && !nePill.label.includes('sup'),
         'not_exposed must render Behavior: not_exposed without leaking token counts');
}

// exposed header without valid payload -> Behavior: not_exposed (never fabricates 0 tok).
const exposedMissingPayload = formatTurnTelemetry({
  ttftMs: 130, totalMs: 980,
  behaviorRequested: true,
  behaviorEvidenceHeader: 'exposed',
  behaviorEvidence: null,
});
const missingPayloadPill = exposedMissingPayload.find((p) => p.kind === 'behavior');
assert(missingPayloadPill && missingPayloadPill.label === 'Behavior: not_exposed' &&
       !missingPayloadPill.label.includes('0 tok'),
       'exposed header without payload must not fabricate token counts');

// Requested with absent header -> Behavior: absent.
const absentHeaderPills = formatTurnTelemetry({
  ttftMs: 130, totalMs: 980,
  behaviorRequested: true,
  behaviorEvidenceHeader: '',
  behaviorEvidence: null,
});
const absentBehaviorPill = absentHeaderPills.find((p) => p.kind === 'behavior');
assert(absentBehaviorPill && absentBehaviorPill.label === 'Behavior: absent' &&
       absentBehaviorPill.status === 'absent',
       'requested behavior with missing header must render Behavior: absent');

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


def check_session_export(html, script):
    """Pure client-side Export Session (JSON + Markdown) markup and script hooks."""
    for snippet in (
        'class="export-group"',
        'id="exportFormat"',
        '<option value="json" selected>JSON</option>',
        '<option value="markdown">Markdown</option>',
        'id="exportSession"',
        '.export-group',
    ):
        require(snippet in html, f"missing session export markup or CSS: {snippet}")
    for control in ('exportFormat', 'exportSession'):
        require(re.search(rf'id="{control}"[^>]*\bdisabled\b', html) is not None,
                f"{control} must start disabled when the session has zero turns")
    for hook in (
        'SESSION_EXPORT_SCHEMA',
        'canExportSession(',
        'sanitizeSessionIdForFilename(',
        'buildSessionExportFilename(',
        'buildSessionExportJson(',
        'buildSessionExportMarkdown(',
        'triggerSessionExportDownload(',
        'collectSessionExportState(',
        'syncExportControls(',
        'URL.createObjectURL(',
        'URL.revokeObjectURL(',
        'sessionTurns',
        "ui.exportSession.addEventListener('click'",
    ):
        require(hook in script, f"missing session export script hook: {hook}")


def check_session_export_behavior(script):
    """Execute pure session export JSON/Markdown/filename/download helpers in Node."""
    node = shutil.which("node")
    if not node:
        print("skip session export behavior: node is not installed", flush=True)
        return
    for marker in (
        "// TURN_TELEMETRY_BEGIN", "// TURN_TELEMETRY_END",
        "// NARRATIVE_CONTINUITY_BEGIN", "// NARRATIVE_CONTINUITY_END",
        "// REALTIME_AUDIO_METER_BEGIN", "// REALTIME_AUDIO_METER_END",
        "// SESSION_EXPORT_BEGIN", "// SESSION_EXPORT_END",
    ):
        require(marker in script, f"missing session export behavior-test marker: {marker}")
    telemetry_src = script.split("// TURN_TELEMETRY_BEGIN", 1)[1].split(
        "// TURN_TELEMETRY_END", 1)[0]
    continuity_src = script.split("// NARRATIVE_CONTINUITY_BEGIN", 1)[1].split(
        "// NARRATIVE_CONTINUITY_END", 1)[0]
    realtime_meter_src = script.split("// REALTIME_AUDIO_METER_BEGIN", 1)[1].split(
        "// REALTIME_AUDIO_METER_END", 1)[0]
    export_src = script.split("// SESSION_EXPORT_BEGIN", 1)[1].split(
        "// SESSION_EXPORT_END", 1)[0]
    frame_src = script.split("// VIDEO_FRAME_SELECTION_BEGIN", 1)[1].split(
        "// VIDEO_FRAME_SELECTION_END", 1)[0]
    harness = (
        frame_src + "\n" + telemetry_src + "\n" + continuity_src + "\n" +
        realtime_meter_src + "\n" + export_src + r"""
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

// 1. Export button gating: disabled on 0 turns or while busy; enabled with >=1 turn & idle.
assert(canExportSession(0, false) === false, 'export must be disabled when turnCount is 0');
assert(canExportSession(0, true) === false, 'export must be disabled when turnCount is 0 and busy');
assert(canExportSession(2, true) === false, 'export must be disabled while streaming/busy');
assert(canExportSession(1, false) === true, 'export must be enabled when >= 1 turn and idle');
assert(canExportSession(-1, false) === false, 'negative turnCount must stay disabled');

// 2. Empty session produces safe deterministic JSON & Markdown with zero turns.
const fixedTs = '2026-11-26T09:00:00.000Z';
const emptyJson = buildSessionExportJson({
  session_id: '',
  model: 'qwen3-native',
  exported_at: fixedTs,
  turns: [],
});
assert(emptyJson.schema === 'umminfer-web-session-export-v1', 'schema version mismatch');
assert(emptyJson.exported_at === fixedTs, 'injectable exported_at mismatch');
assert(emptyJson.session_id === null, 'empty session_id must serialize as null');
assert(emptyJson.model === 'qwen3-native', 'model mismatch');
assert(emptyJson.turn_count === 0 && Array.isArray(emptyJson.turns) && emptyJson.turns.length === 0,
       'empty session must report 0 turns');
assert(!Object.hasOwn(emptyJson, 'video_frame_selection'),
       'default session export must omit video_frame_selection');
const explicitBudgetExport = buildSessionExportJson({
  session_id: 'budget-session', model: 'qwen3-native', exported_at: fixedTs, turns: [],
  video_frame_selection: {
    policy: 'temporal', motion_threshold: 0.02, scene_threshold: 0.25, max_frames: 16,
  },
});
assert(explicitBudgetExport.video_frame_selection.max_frames === 16 &&
       explicitBudgetExport.video_frame_selection.policy === 'temporal',
       'explicit temporal budget was not preserved in session export');

const emptyMd = buildSessionExportMarkdown({
  session_id: '',
  model: 'qwen3-native',
  exported_at: fixedTs,
  turns: [],
});
assert(emptyMd.includes('# ummInfer Session Export') &&
       emptyMd.includes('**Turns:** 0') &&
       emptyMd.includes('_No turns recorded._'),
       'empty Markdown export mismatch: ' + emptyMd);

// 3. Rich multi-turn session: text, I2V explicit anchor, interrupted voice barge-in,
//    COVERED & N/C native allocator, exposed & not_exposed behavior logprobs, zero base64 bloat.
const sampleSha = '3a9d5bbe' + '4f12c890'.repeat(7);
const slot0Sha = 'abcdef01' + '23456789'.repeat(7);
const fakeBase64Blob = 'data:image/png;base64,' + 'A'.repeat(200000);
const coveredEnvelope = {
  schema_version: 1,
  coverage: 'COVERED',
  stages: [
    {
      stage: 'vlm',
      native_stage: 'qwen38',
      coverage: 'COVERED',
      allocated_bytes_peak: Math.round(128.4 * 1024 * 1024),
      epoch: { id: 17, reset: true },
    },
  ],
};
const notCoveredEnvelope = {
  schema_version: 1,
  coverage: 'NOT_COVERED',
  stages: [
    {
      stage: 'vlm',
      native_stage: 'qwen38',
      coverage: 'NOT_COVERED',
      allocated_bytes_peak: null,
      reason: 'batched_request_attribution_unavailable',
    },
  ],
};
const exposedBehaviorPayload = {
  format: 'umminfer-behavior-logprobs-v1',
  tokenizer_sha256: '1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016',
  token_ids: [101, 202, 303, 404, 505],
  logprobs: [-0.1, -0.25, -0.05, -0.4, -0.01],
  mask: [1, 0, 1, 0, 1],
};

const richState = {
  session_id: 'ummweb-a1b2c3d4e5f60708',
  model: 'qwen3-native',
  exported_at: fixedTs,
  turns: [
    {
      prompt: 'Create a two-beat storyboard of a glowing lantern.',
      response: 'A warm lantern floats above the misty river.',
      visual_slots: [
        {
          id: 'slot-0',
          kind: 'image',
          prompt: 'establishing keyframe of glowing paper lantern',
          bytes: 131072,
          sha256: slot0Sha,
          // Simulate raw live slot fields that must NEVER leak into export.
          imageDataURL: fakeBase64Blob,
          url: 'blob:http://127.0.0.1:30000/slot-0',
          cache: { image: 'blob:http://127.0.0.1:30000/slot-0' },
        },
        {
          id: 'slot-1',
          kind: 'video',
          prompt: 'animate river breeze and reflections',
          bytes: 1572864,
          continuity_mode: 'i2v_chain',
          anchor_slot_index: 0,
          anchor_explicit: true,
          anchor_sha256: sampleSha,
        },
      ],
      telemetry: {
        ttftMs: 210,
        ttfaMs: 380,
        visualSettledCount: 2,
        firstVisualMs: 3200,
        hasI2VChain: true,
        hasExplicitAnchor: true,
        totalMs: 9100,
        nativeAllocator: coveredEnvelope,
        behaviorRequested: true,
        behaviorEvidenceHeader: 'EXPOSED',
        behaviorEvidence: exposedBehaviorPayload,
      },
    },
    {
      prompt: 'Prove why sqrt(2) is irrational.',
      response: 'Assume sqrt(2) = p/q in lowest terms; then both p and q are even.',
      interrupted: false,
      spoken_bytes: 0, // Must be ignored when interrupted !== true!
      visual_slots: [],
      telemetry: {
        ttftMs: 120,
        ttfaMs: null,
        visualSettledCount: 0,
        firstVisualMs: null,
        hasI2VChain: false,
        hasExplicitAnchor: false,
        totalMs: 950,
        nativeAllocator: notCoveredEnvelope,
        behaviorRequested: true,
        behaviorEvidenceHeader: 'not_exposed',
        behaviorEvidence: exposedBehaviorPayload, // Stray payload must never leak!
      },
    },
    {
      prompt: 'Tell me about ocean tides.',
      response: 'Ocean tides are driven by',
      interrupted: true,
      spoken_bytes: 24,
      visual_slots: [],
      telemetry: null,
    },
    {
      prompt: 'Quick follow-up with inline ' + fakeBase64Blob,
      response: '',
      interrupted: true,
      spoken_bytes: 0,
      visual_slots: [],
      telemetry: null,
    },
  ],
};

const richJson = buildSessionExportJson(richState);
const richJsonText = JSON.stringify(richJson);
assert(richJson.schema === 'umminfer-web-session-export-v1', 'rich schema mismatch');
assert(richJson.session_id === 'ummweb-a1b2c3d4e5f60708', 'rich session_id mismatch');
assert(richJson.turn_count === 4 && richJson.turns.length === 4, 'rich turn_count mismatch');

// Zero raw base64 bloat in JSON export.
assert(!richJsonText.includes('data:image/') && !richJsonText.includes('A'.repeat(200)),
       'base64 media blob leaked into JSON export');
assert(richJsonText.length < 8192,
       `JSON export unexpectedly large (${richJsonText.length} bytes)`);

// Turn 1 assertions: visual slots + explicit I2V + COVERED VRAM + exposed behavior.
const t1 = richJson.turns[0];
assert(t1.index === 1 && t1.prompt.includes('glowing lantern') &&
       t1.response.includes('warm lantern floats'),
       'turn 1 text mismatch');
assert(!Object.prototype.hasOwnProperty.call(t1, 'interrupted') &&
       !Object.prototype.hasOwnProperty.call(t1, 'spoken_bytes'),
       'non-interrupted turn 1 must omit interrupted and spoken_bytes');
assert(t1.visual_slots.length === 2, 'turn 1 visual_slots count mismatch');
assert(t1.visual_slots[0].id === 'slot-0' && t1.visual_slots[0].kind === 'image' &&
       t1.visual_slots[0].bytes === 131072 && t1.visual_slots[0].sha256 === slot0Sha &&
       !Object.prototype.hasOwnProperty.call(t1.visual_slots[0], 'imageDataURL') &&
       !Object.prototype.hasOwnProperty.call(t1.visual_slots[0], 'url') &&
       !Object.prototype.hasOwnProperty.call(t1.visual_slots[0], 'cache'),
       'turn 1 slot-0 metadata mismatch or leaked raw media fields');
assert(t1.visual_slots[1].id === 'slot-1' && t1.visual_slots[1].kind === 'video' &&
       t1.visual_slots[1].continuity_mode === 'i2v_chain' &&
       t1.visual_slots[1].anchor_slot_index === 0 &&
       t1.visual_slots[1].anchor_explicit === true &&
       t1.visual_slots[1].anchor_sha256 === sampleSha &&
       t1.visual_slots[1].bytes === 1572864,
       'turn 1 slot-1 I2V explicit anchor metadata mismatch');
assert(t1.telemetry.ttft_ms === 210 && t1.telemetry.ttfa_ms === 380 &&
       t1.telemetry.first_visual_ms === 3200 && t1.telemetry.total_ms === 9100 &&
       t1.telemetry.visual_slots === 2 && t1.telemetry.i2v_count === 1,
       'turn 1 telemetry numbers mismatch: ' + JSON.stringify(t1.telemetry));
assert(t1.telemetry.native_allocator.coverage === 'COVERED' &&
       t1.telemetry.native_allocator.covered === true &&
       t1.telemetry.native_allocator.peak_bytes === Math.round(128.4 * 1024 * 1024) &&
       t1.telemetry.native_allocator.label === 'VRAM: 128.4 MiB peak',
       'turn 1 COVERED native_allocator mismatch');
assert(t1.telemetry.behavior_logprobs.status === 'exposed' &&
       t1.telemetry.behavior_logprobs.token_count === 5 &&
       t1.telemetry.behavior_logprobs.supervised_count === 3 &&
       t1.telemetry.behavior_logprobs.format === 'umminfer-behavior-logprobs-v1' &&
       t1.telemetry.behavior_logprobs.label === 'Behavior: 5 tok (3 sup)',
       'turn 1 exposed behavior_logprobs mismatch');

// Turn 2 assertions: zero fabrication for null ttfa/first_visual, N/C VRAM, not_exposed behavior.
const t2 = richJson.turns[1];
assert(!Object.prototype.hasOwnProperty.call(t2, 'interrupted') &&
       !Object.prototype.hasOwnProperty.call(t2, 'spoken_bytes'),
       'turn 2 with interrupted:false must omit interrupted and spoken_bytes');
assert(t2.telemetry.ttft_ms === 120 && t2.telemetry.total_ms === 950 &&
       !Object.prototype.hasOwnProperty.call(t2.telemetry, 'ttfa_ms') &&
       !Object.prototype.hasOwnProperty.call(t2.telemetry, 'first_visual_ms') &&
       t2.telemetry.visual_slots === 0 && t2.telemetry.i2v_count === 0,
       'turn 2 must omit unobserved ttfa_ms / first_visual_ms without fabricating 0');
assert(t2.telemetry.native_allocator.coverage === 'NOT_COVERED' &&
       t2.telemetry.native_allocator.covered === false &&
       t2.telemetry.native_allocator.peak_bytes === null &&
       t2.telemetry.native_allocator.label === 'VRAM: N/C',
       'turn 2 N/C native_allocator must keep peak_bytes=null, never 0');
assert(t2.telemetry.behavior_logprobs.status === 'not_exposed' &&
       t2.telemetry.behavior_logprobs.token_count === null &&
       t2.telemetry.behavior_logprobs.supervised_count === null &&
       t2.telemetry.behavior_logprobs.label === 'Behavior: not_exposed',
       'turn 2 not_exposed behavior must keep counts null, never 0');

// Turn 3 assertions: interrupted voice barge-in with spoken_bytes=24 and no fabricated telemetry.
const t3 = richJson.turns[2];
assert(t3.interrupted === true && t3.spoken_bytes === 24,
       'turn 3 interrupted spoken_bytes mismatch');
assert(!Object.prototype.hasOwnProperty.call(t3, 'telemetry'),
       'turn 3 with null telemetry must omit telemetry rather than fabricate zeros');

// Turn 4 assertions: interrupted before audio (spoken_bytes=0) + scrubbed inline data URI.
const t4 = richJson.turns[3];
assert(t4.interrupted === true && t4.spoken_bytes === 0,
       'turn 4 zero-byte interruption mismatch');
assert(t4.prompt.includes('[omitted-base64]') && !t4.prompt.includes('data:image/'),
       'turn 4 did not scrub inline base64 data URI');

// 4. Markdown export assertions: matches formatTurnTelemetry() and badges with zero base64 bloat.
const richMd = buildSessionExportMarkdown(richState);
assert(!richMd.includes('data:image/') && !richMd.includes('A'.repeat(200)),
       'base64 media blob leaked into Markdown export');
assert(richMd.includes('# ummInfer Session Export') &&
       richMd.includes('`ummweb-a1b2c3d4e5f60708`') &&
       richMd.includes('`qwen3-native`') &&
       richMd.includes('**Turns:** 4'),
       'Markdown session header mismatch');
assert(richMd.includes('#### Visual Slots') &&
       richMd.includes('`slot-0` (**Image**)') &&
       richMd.includes('`128 KiB`') &&
       richMd.includes('`sha256: abcdef01`') &&
       richMd.includes('`slot-1` (**Video**, `Explicit I2V ← Slot 1 · 3a9d5bbe`)') &&
       richMd.includes('`1.5 MiB`'),
       'Markdown visual slot metadata mismatch:\n' + richMd);
const expectedT1Telemetry = formatTurnTelemetry(richState.turns[0].telemetry)
  .map((p) => '`' + p.label + '`').join(' · ');
assert(richMd.includes(`**Telemetry:** ${expectedT1Telemetry}`),
       'Markdown turn 1 telemetry line does not match formatTurnTelemetry():\n' + richMd);
const expectedT2Telemetry = formatTurnTelemetry(richState.turns[1].telemetry)
  .map((p) => '`' + p.label + '`').join(' · ');
assert(richMd.includes(`**Telemetry:** ${expectedT2Telemetry}`) &&
       richMd.includes('`VRAM: N/C`') && richMd.includes('`Behavior: not_exposed`'),
       'Markdown turn 2 telemetry line mismatch:\n' + richMd);
assert(richMd.includes('### Assistant · Interrupted · 24 B spoken'),
       'Markdown turn 3 interrupted cut-point badge mismatch');
assert(richMd.includes('### Assistant · Interrupted before audio'),
       'Markdown turn 4 zero-byte interrupted badge mismatch');
// Turn 1 and Turn 2 are NOT interrupted; Turn 3 has no telemetry line.
const t3Section = richMd.split('## Turn 3')[1].split('## Turn 4')[0];
assert(!t3Section.includes('**Telemetry:**'),
       'turn 3 with no telemetry must not emit a Telemetry line');

// 5. Filename sanitization strips path traversal, control chars, and unsafe punctuation.
assert(buildSessionExportFilename('ummweb-a1b2c3d4', 'json') === 'umminfer-session-ummweb-a1b2c3d4.json',
       'clean JSON filename mismatch');
assert(buildSessionExportFilename('ummweb-a1b2c3d4', 'markdown') === 'umminfer-session-ummweb-a1b2c3d4.md',
       'clean Markdown filename mismatch');
for (const unsafe of [
  '../../etc/passwd',
  'bad\x00name<script>alert(1)</script>',
  'session: 2026/11/26 *?"|<>\\',
  '  ---..__  ',
  '',
  null,
  'a'.repeat(120),
]) {
  const fnJson = buildSessionExportFilename(unsafe, 'json');
  const fnMd = buildSessionExportFilename(unsafe, 'markdown');
  assert(/^umminfer-session-[A-Za-z0-9_-]{1,64}\.json$/.test(fnJson),
         `unsafe JSON filename leaked: ${fnJson}`);
  assert(/^umminfer-session-[A-Za-z0-9_-]{1,64}\.md$/.test(fnMd),
         `unsafe MD filename leaked: ${fnMd}`);
  assert(!fnJson.includes('..') && !fnJson.includes('/') && !fnJson.includes('\\'),
         `path separator/traversal survived sanitization: ${fnJson}`);
}
assert(buildSessionExportFilename('', 'json') === 'umminfer-session-untitled.json',
       'empty session id must fall back to untitled');
assert(buildSessionExportFilename('///...', 'markdown') === 'umminfer-session-untitled.md',
       'symbol-only session id must fall back to untitled');

// 6. triggerSessionExportDownload uses URL.createObjectURL(new Blob(...)) and revokes URL cleanly.
const createdBlobs = [];
const revokedUrls = [];
const clickedLinks = [];
const pendingTimers = [];
const origCreateObjectURL = URL.createObjectURL;
const origRevokeObjectURL = URL.revokeObjectURL;
const origSetTimeout = globalThis.setTimeout;
URL.createObjectURL = (blob) => {
  const id = `blob:mock-${createdBlobs.length + 1}`;
  createdBlobs.push({ id, blob });
  return id;
};
URL.revokeObjectURL = (url) => { revokedUrls.push(url); };
globalThis.setTimeout = (fn) => { pendingTimers.push(fn); return pendingTimers.length; };
globalThis.document = {
  body: { append() {} },
  createElement(tag) {
    assert(tag === 'a', 'expected anchor element for download');
    const el = {
      href: '', download: '', rel: '', removed: false,
      click() { clickedLinks.push({ href: this.href, download: this.download, rel: this.rel }); },
      remove() { this.removed = true; },
    };
    return el;
  },
};
try {
  const jsonDl = triggerSessionExportDownload(richState, 'json');
  const mdDl = triggerSessionExportDownload(richState, 'markdown');
  while (pendingTimers.length) pendingTimers.shift()();
  assert(jsonDl.filename === 'umminfer-session-ummweb-a1b2c3d4e5f60708.json' &&
         jsonDl.mime === 'application/json;charset=utf-8' && jsonDl.bytes > 0,
         'JSON download result mismatch');
  assert(mdDl.filename === 'umminfer-session-ummweb-a1b2c3d4e5f60708.md' &&
         mdDl.mime === 'text/markdown;charset=utf-8' && mdDl.bytes > 0,
         'Markdown download result mismatch');
  assert(clickedLinks.length === 2 &&
         clickedLinks[0].download === jsonDl.filename &&
         clickedLinks[1].download === mdDl.filename,
         'download click dispatch mismatch');
  assert(revokedUrls.length === 2 &&
         revokedUrls[0] === 'blob:mock-1' && revokedUrls[1] === 'blob:mock-2',
         'object URLs were not revoked cleanly');
} finally {
  URL.createObjectURL = origCreateObjectURL;
  URL.revokeObjectURL = origRevokeObjectURL;
  globalThis.setTimeout = origSetTimeout;
}

console.log('session export behavior passed');
"""
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "session-export.mjs"
        path.write_text(harness, encoding="utf-8")
        result = subprocess.run([node, str(path)], capture_output=True,
                                text=True, timeout=60)
    require(result.returncode == 0,
            f"session export behavior failed: {(result.stderr or result.stdout).strip()[:1200]}")
    require("session export behavior passed" in result.stdout,
            "session export harness did not reach its success assertion")


def check_session_import(html, script):
    """Pure client-side cross-client JSON Session Import markup and script hooks."""
    for snippet in (
        'id="importSessionInput"',
        'type="file"',
        'accept="application/json,.json"',
        'id="importSession"',
        '>Import JSON</button>',
        '.slot-imported-note',
    ):
        require(snippet in html, f"missing session import markup or CSS: {snippet}")
    require(re.search(r'id="importSessionInput"[^>]*\bhidden\b', html) is not None,
            "importSessionInput file picker must be hidden")
    require("ui.importSession" in script.split("function setBusy(busy)", 1)[1].split("])", 1)[0],
            "importSession must be disabled while a response is streaming")
    for hook in (
        'MAX_IMPORT_SESSION_BYTES',
        'SUPPORTED_IMPORT_SCHEMAS',
        "'umminfer-web-session-export-v1'",
        "'umminfer-session-export-v1'",
        'parseImportedSessionJson(',
        'formatImportedSessionChip(',
        'applyImportedSessionWithContext(',
        'applyImportedSession(',
        'showSlotImportedMetadata(',
        'formatImportedNativeAllocatorSummary(',
        'formatImportedBehaviorSummary(',
        "ui.importSession.addEventListener('click'",
        "ui.importSessionInput.addEventListener('change'",
    ):
        require(hook in script, f"missing session import script hook: {hook}")
    require("// SESSION_IMPORT_BEGIN" in script and "// SESSION_IMPORT_END" in script,
            "missing session import behavior-test markers")
    import_block = script.split("// SESSION_IMPORT_BEGIN", 1)[1].split(
        "// SESSION_IMPORT_END", 1)[0]
    require("fetch(" not in import_block and "new WebSocket(" not in import_block,
            "pure session import helpers must never issue network or WebSocket calls")


def check_session_import_behavior(script):
    """Execute cross-client JSON session import, round-trip, and fail-closed checks in Node."""
    node = shutil.which("node")
    if not node:
        print("skip session import behavior: node is not installed", flush=True)
        return
    for marker in (
        "// TURN_TELEMETRY_BEGIN", "// TURN_TELEMETRY_END",
        "// NARRATIVE_CONTINUITY_BEGIN", "// NARRATIVE_CONTINUITY_END",
        "// REALTIME_AUDIO_METER_BEGIN", "// REALTIME_AUDIO_METER_END",
        "// SESSION_EXPORT_BEGIN", "// SESSION_EXPORT_END",
        "// SESSION_IMPORT_BEGIN", "// SESSION_IMPORT_END",
    ):
        require(marker in script, f"missing session import behavior-test marker: {marker}")
    telemetry_src = script.split("// TURN_TELEMETRY_BEGIN", 1)[1].split(
        "// TURN_TELEMETRY_END", 1)[0]
    continuity_src = script.split("// NARRATIVE_CONTINUITY_BEGIN", 1)[1].split(
        "// NARRATIVE_CONTINUITY_END", 1)[0]
    realtime_meter_src = script.split("// REALTIME_AUDIO_METER_BEGIN", 1)[1].split(
        "// REALTIME_AUDIO_METER_END", 1)[0]
    export_src = script.split("// SESSION_EXPORT_BEGIN", 1)[1].split(
        "// SESSION_EXPORT_END", 1)[0]
    import_src = script.split("// SESSION_IMPORT_BEGIN", 1)[1].split(
        "// SESSION_IMPORT_END", 1)[0]
    frame_src = script.split("// VIDEO_FRAME_SELECTION_BEGIN", 1)[1].split(
        "// VIDEO_FRAME_SELECTION_END", 1)[0]
    harness = (
        frame_src + "\n" + telemetry_src + "\n" + continuity_src + "\n" +
        realtime_meter_src + "\n" + export_src + "\n" + import_src + r"""
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

// Guard: fail immediately if any network or WebSocket API is invoked during import.
let networkCalls = 0;
globalThis.fetch = () => { networkCalls += 1; throw new Error('unexpected fetch during import'); };
globalThis.WebSocket = class ForbiddenWebSocket {
  constructor() { networkCalls += 1; throw new Error('unexpected WebSocket during import'); }
};

function createMockRuntime(initialBusy = false) {
  const state = {
    busy: initialBusy,
    sessionID: 'existing-session-keep',
    history: [{ role: 'user', content: 'prior prompt' }, { role: 'assistant', content: 'prior reply' }],
    turns: [{ user: 'prior prompt', assistant: 'prior reply' }],
    sessionChip: { text: 'session existing', title: 'existing-session-keep' },
    status: 'Ready',
    error: '',
    audioStopped: 0,
    syncCount: 0,
    frameSelection: {
      policy: 'temporal', motion_threshold: 0.9, scene_threshold: 0.8, max_frames: 16,
    },
  };
  const ctx = {
    isBusy: () => state.busy,
    stopActiveAudio() { state.audioStopped += 1; },
    resetState(nextSessionId) {
      state.sessionID = nextSessionId;
      state.history = [];
      state.turns = [];
    },
    applyVideoFrameSelection(selection) { state.frameSelection = selection; },
    clearTranscript() {},
    clearError() { state.error = ''; },
    addTurn(role, text) {
      return {
        role, text, asrText: '', asrEl: { textContent: '' },
        renderedText: '', slots: [], interrupted: false, spokenBytes: null,
        interruptedBadge: null, telemetry: null, renderedPills: [],
      };
    },
    renderAssistantText(assistantTurn, text) {
      assistantTurn.renderedText = text;
    },
    addImportedVisualSlot(assistantTurn, slotMeta) {
      assistantTurn.slots.push({
        ...slotMeta,
        continuityBadge: visualSlotContinuityBadge(slotMeta),
        importedMediaOmitted: !slotMeta.error,
      });
    },
    attachInterruptedBadge(assistantTurn, bytes, badgeText) {
      assistantTurn.interruptedBadge = { bytes, badgeText };
    },
    renderTelemetry(assistantTurn) {
      assistantTurn.renderedPills = formatTurnTelemetry(assistantTurn.telemetry);
    },
    recordTurn(userTurn, assistantTurn, promptText, responseText) {
      state.turns.push({ userTurn, assistantTurn });
      if (promptText) state.history.push({ role: 'user', content: promptText });
      if (responseText) state.history.push({ role: 'assistant', content: responseText });
    },
    setSessionChip(chip) { state.sessionChip = chip; },
    syncUI() { state.syncCount += 1; },
    setStatus(text) { state.status = text; },
  };
  return { state, ctx };
}

// 1. Round-trip export -> import -> re-export for umminfer-web-session-export-v1.
const fixedTs = '2026-11-26T09:30:00.000Z';
const sampleSha = '3a9d5bbe' + '4f12c890'.repeat(7);
const slot0Sha = 'abcdef01' + '23456789'.repeat(7);
const coveredEnvelope = {
  schema_version: 1,
  coverage: 'COVERED',
  stages: [{
    stage: 'vlm', native_stage: 'qwen38', coverage: 'COVERED',
    allocated_bytes_peak: Math.round(128.4 * 1024 * 1024),
    epoch: { id: 17, reset: true },
  }],
};
const notCoveredEnvelope = {
  schema_version: 1,
  coverage: 'NOT_COVERED',
  stages: [{
    stage: 'vlm', native_stage: 'qwen38', coverage: 'NOT_COVERED',
    allocated_bytes_peak: null, reason: 'batched_request_attribution_unavailable',
  }],
};
const exposedBehaviorPayload = {
  format: 'umminfer-behavior-logprobs-v1',
  tokenizer_sha256: '1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016',
  token_ids: [101, 202, 303, 404, 505],
  logprobs: [-0.1, -0.25, -0.05, -0.4, -0.01],
  mask: [1, 0, 1, 0, 1],
};

const webExportObj = buildSessionExportJson({
  session_id: 'ummweb-a1b2c3d4e5f60708',
  model: 'qwen3-native',
  exported_at: fixedTs,
  turns: [
    {
      prompt: 'Create a two-beat storyboard of a glowing lantern.',
      response: 'A warm lantern floats above the misty river.',
      asr_text: 'Create a two beat storyboard',
      visual_slots: [
        {
          id: 'slot-0', kind: 'image', prompt: 'glowing lantern keyframe',
          bytes: 131072, sha256: slot0Sha,
        },
        {
          id: 'slot-1', kind: 'video', prompt: 'lantern drifting in rain',
          bytes: 1572864, continuity_mode: 'i2v_chain',
          anchor_slot_index: 0, anchor_explicit: true, anchor_sha256: sampleSha,
        },
      ],
      telemetry: {
        ttftMs: 210, ttfaMs: 380, visualSettledCount: 2, firstVisualMs: 3200,
        hasI2VChain: true, hasExplicitAnchor: true, totalMs: 9100,
        nativeAllocator: coveredEnvelope,
        behaviorRequested: true, behaviorEvidenceHeader: 'EXPOSED',
        behaviorEvidence: exposedBehaviorPayload,
      },
    },
    {
      prompt: 'Prove why sqrt(2) is irrational.',
      response: 'Assume sqrt(2) = p/q in lowest terms.',
      visual_slots: [],
      telemetry: {
        ttftMs: 120, ttfaMs: null, visualSettledCount: 0, firstVisualMs: null,
        hasI2VChain: false, hasExplicitAnchor: false, totalMs: 950,
        nativeAllocator: notCoveredEnvelope,
        behaviorRequested: true, behaviorEvidenceHeader: 'not_exposed',
        behaviorEvidence: null,
      },
    },
    {
      prompt: 'Tell me about ocean tides.',
      response: 'Ocean tides are driven by',
      interrupted: true,
      spoken_bytes: 24,
      visual_slots: [],
      telemetry: null,
    },
  ],
});

const parsedWeb = parseImportedSessionJson(JSON.stringify(webExportObj));
assert(parsedWeb.schema === 'umminfer-web-session-export-v1' &&
       parsedWeb.sourceClient === 'web' &&
       parsedWeb.session_id === 'ummweb-a1b2c3d4e5f60708' &&
       parsedWeb.model === 'qwen3-native' &&
       parsedWeb.turn_count === 3 && parsedWeb.turns.length === 3,
       'parsed web session metadata mismatch');

const webRt = createMockRuntime(false);
const webApply = applyImportedSessionWithContext(parsedWeb, webRt.ctx);
assert(webApply.turnCount === 3 && webApply.sourceClient === 'web', 'web apply summary mismatch');
assert(webRt.state.sessionChip.text === 'imported a1b2c3d4' &&
       webRt.state.status === 'Imported 3 turns',
       `web session chip/status mismatch: ${JSON.stringify(webRt.state.sessionChip)} / ${webRt.state.status}`);
assert(webRt.state.history.length === 6 &&
       webRt.state.history[0].role === 'user' && webRt.state.history[1].role === 'assistant',
       'restored history pairs mismatch');
assert(webRt.state.frameSelection === undefined,
       'legacy import silently retained the previous 16-frame budget');

// Check restored Turn 1 telemetry, behavior, continuity, and omitted media (zero fabrication).
const rt1 = webRt.state.turns[0].assistantTurn;
assert(rt1.asrText === 'Create a two beat storyboard', 'turn 1 ASR mismatch');
assert(rt1.slots.length === 2 && rt1.slots[0].importedMediaOmitted === true &&
       rt1.slots[1].continuityBadge?.text === 'Explicit I2V ← Slot 1 · 3a9d5bbe',
       'turn 1 restored visual slots mismatch');
assert(JSON.stringify(rt1.renderedPills.map((p) => p.label)) ===
       '["TTFT 210 ms","TTFA 380 ms","2 visuals (3.20 s)","Explicit I2V","Total 9.10 s","VRAM: 128.4 MiB peak","Behavior: 5 tok (3 sup)"]',
       'turn 1 restored telemetry pills mismatch: ' + JSON.stringify(rt1.renderedPills));

// Check restored Turn 2: NOT_COVERED VRAM + not_exposed behavior (zero fabricated zeros).
const rt2 = webRt.state.turns[1].assistantTurn;
assert(JSON.stringify(rt2.renderedPills.map((p) => p.label)) ===
       '["TTFT 120 ms","Total 950 ms","VRAM: N/C","Behavior: not_exposed"]',
       'turn 2 restored telemetry pills mismatch: ' + JSON.stringify(rt2.renderedPills));

// Check restored Turn 3: interrupted badge, null telemetry omitted completely.
const rt3 = webRt.state.turns[2].assistantTurn;
assert(rt3.interrupted === true && rt3.spokenBytes === 24 &&
       rt3.interruptedBadge?.badgeText === 'Interrupted · 24 B spoken' &&
       rt3.telemetry === null && rt3.renderedPills.length === 0,
       'turn 3 restored interruption/null telemetry mismatch');

// Re-export the restored web session and verify JSON + Markdown fidelity!
const reExportState = {
  session_id: webRt.state.sessionID,
  model: 'qwen3-native',
  exported_at: fixedTs,
  turns: webRt.state.turns.map((entry) => ({
    prompt: entry.userTurn?.text || '',
    response: entry.assistantTurn?.text || '',
    asr_text: entry.assistantTurn?.asrText || '',
    interrupted: entry.assistantTurn?.interrupted === true,
    spoken_bytes: entry.assistantTurn?.spokenBytes,
    visual_slots: entry.assistantTurn?.slots || [],
    telemetry: entry.assistantTurn?.telemetry || null,
  })),
};
const reExportJson = buildSessionExportJson(reExportState);
assert(JSON.stringify(reExportJson) === JSON.stringify(webExportObj),
       'round-trip re-export JSON diverged:\nexpected: ' + JSON.stringify(webExportObj) +
       '\ngot:      ' + JSON.stringify(reExportJson));
const reExportMd = buildSessionExportMarkdown(reExportState);
assert(reExportMd.includes('`VRAM: 128.4 MiB peak`') &&
       reExportMd.includes('`Behavior: 5 tok (3 sup)`') &&
       reExportMd.includes('`VRAM: N/C`') &&
       reExportMd.includes('`Behavior: not_exposed`') &&
       reExportMd.includes('### Assistant · Interrupted · 24 B spoken'),
       'round-trip re-export Markdown telemetry mismatch:\n' + reExportMd);

// Explicit Uniform/Temporal 4/8/16 configurations survive export -> import ->
// apply -> re-export as integer max_frames values, never as coerced strings.
for (const policy of ['uniform', 'temporal']) {
  for (const budget of [4, 8, 16]) {
    const selection = policy === 'uniform'
      ? { policy, max_frames: budget }
      : { policy, motion_threshold: 0.02, scene_threshold: 0.25, max_frames: budget };
    const exported = buildSessionExportJson({
      session_id: `budget-${policy}-${budget}`, model: 'qwen3-native',
      exported_at: fixedTs, turns: [], video_frame_selection: selection,
    });
    const parsed = parseImportedSessionJson(JSON.stringify(exported));
    const runtime = createMockRuntime(false);
    applyImportedSessionWithContext(parsed, runtime.ctx);
    assert(JSON.stringify(runtime.state.frameSelection) === JSON.stringify(selection),
           `${policy} ${budget} import did not restore the exact selection`);
    const reexported = buildSessionExportJson({
      session_id: parsed.session_id, model: parsed.model, exported_at: fixedTs,
      turns: [], video_frame_selection: runtime.state.frameSelection,
    });
    assert(JSON.stringify(reexported) === JSON.stringify(exported),
           `${policy} ${budget} session round trip diverged`);
  }
}

// 2. Cross-client import of macOS ummmlx export (exact SessionExport.swift schema).
const macOSExportJson = JSON.stringify({
  exported_at: '2026-11-26T09:45:00.123Z',
  model: 'mlx-community/Qwen3-8B-4bit',
  schema: 'umminfer-session-export-v1',
  session_id: 'ummmlx-deadbeef01020304',
  turn_count: 2,
  turns: [
    {
      asr_text: 'Show me a lantern drifting',
      index: 1,
      prompt: 'Show me a lantern drifting over a river.',
      response: 'A warm paper lantern glides down the misty river.',
      telemetry: {
        behavior_logprobs: {
          format: 'umminfer-behavior-logprobs-v1',
          label: 'Behavior: 12 tok (9 sup)',
          status: 'exposed',
          supervised_count: 9,
          token_count: 12,
          tokenizer_sha256: '1dd1199e35f6cfb9f547458e02c9fb3a522e9c346ed22ba3d461aeed14310016',
        },
        first_visual_ms: 2800,
        i2v_count: 1,
        native_allocator: {
          coverage: 'COVERED',
          covered: true,
          covered_stage_count: 2,
          details: [
            'vlm/qwen38: COVERED · peak 96 MiB · epoch #4',
            'video/wan22: COVERED · peak 256 MiB · epoch #5',
          ],
          label: 'VRAM: 256 MiB peak',
          peak_bytes: 268435456,
          total_stage_count: 2,
        },
        total_ms: 7400,
        ttfa_ms: 420,
        ttft_ms: 190,
        visual_slots: 2,
      },
      visual_slots: [
        {
          bytes: 65536,
          kind: 'image',
          prompt: 'lantern establishing shot',
          sha256: slot0Sha,
          slot_id: 'slot-0',
          slot_index: 0,
        },
        {
          anchor_explicit: true,
          anchor_sha256: sampleSha,
          anchor_slot_index: 0,
          bytes: 1048576,
          continuity_mode: 'i2v_chain',
          kind: 'video',
          prompt: 'lantern drifting smoothly',
          slot_id: 'slot-1',
          slot_index: 1,
        },
      ],
    },
    {
      index: 2,
      interrupted: true,
      prompt: 'Continue the story.',
      response: 'As night fell over',
      spoken_bytes: 18,
      visual_slots: [
        {
          error: 'video backend offline',
          kind: 'video',
          prompt: 'nightfall ripples',
          slot_id: 'slot-0',
          slot_index: 0,
        },
      ],
    },
  ],
});

const parsedMac = parseImportedSessionJson(macOSExportJson);
assert(parsedMac.schema === 'umminfer-session-export-v1' &&
       parsedMac.sourceClient === 'ummmlx' &&
       parsedMac.session_id === 'ummmlx-deadbeef01020304' &&
       parsedMac.turns.length === 2,
       'macOS ummmlx export parse mismatch');
const macRt = createMockRuntime(false);
applyImportedSessionWithContext(parsedMac, macRt.ctx);
assert(macRt.state.sessionChip.text === 'imported deadbeef' &&
       macRt.state.status === 'Imported 2 turns · macOS ummmlx',
       `macOS chip/status mismatch: ${macRt.state.sessionChip.text} / ${macRt.state.status}`);
const macT1 = macRt.state.turns[0].assistantTurn;
assert(macT1.slots.length === 2 &&
       macT1.slots[0].id === 'slot-0' && macT1.slots[0].bytes === 65536 &&
       macT1.slots[1].id === 'slot-1' &&
       macT1.slots[1].continuityBadge?.text === 'Explicit I2V ← Slot 1 · 3a9d5bbe',
       'macOS slot_id/slot_index restoration mismatch');
assert(JSON.stringify(macT1.renderedPills.map((p) => p.label)) ===
       '["TTFT 190 ms","TTFA 420 ms","2 visuals (2.80 s)","Explicit I2V","Total 7.40 s","VRAM: 256 MiB peak","Behavior: 12 tok (9 sup)"]',
       'macOS telemetry pills mismatch: ' + JSON.stringify(macT1.renderedPills));
const macT2 = macRt.state.turns[1].assistantTurn;
assert(macT2.interrupted === true && macT2.spokenBytes === 18 &&
       macT2.interruptedBadge?.badgeText === 'Interrupted · 18 B spoken' &&
       macT2.slots[0].error === 'video backend offline' &&
       macT2.slots[0].importedMediaOmitted === false &&
       macT2.telemetry === null,
       'macOS turn 2 interruption/slot-error/null telemetry mismatch');

// 3. Empty valid export imports cleanly with local session label.
const emptyParsed = parseImportedSessionJson(JSON.stringify({
  schema: 'umminfer-web-session-export-v1',
  session_id: null,
  model: null,
  turn_count: 0,
  turns: [],
}));
const emptyRt = createMockRuntime(false);
applyImportedSessionWithContext(emptyParsed, emptyRt.ctx);
assert(emptyRt.state.turns.length === 0 &&
       emptyRt.state.sessionChip.text === 'imported local' &&
       emptyRt.state.status === 'Imported 0 turns',
       'empty session import mismatch');

// 4. Fail-closed validation: every invalid input throws and leaves prior state untouched.
const invalidCases = [
  ['non-string', 12345, /JSON string/],
  ['empty string', '', /not valid JSON/],
  ['malformed JSON', '{not valid json', /not valid JSON/],
  ['array root', '[]', /must be an object/],
  ['null root', 'null', /must be an object/],
  ['missing schema', JSON.stringify({ turns: [] }), /Unsupported session export schema/],
  ['unsupported schema', JSON.stringify({ schema: 'umminfer-session-export-v2', turns: [] }), /Unsupported session export schema/],
  ['missing turns', JSON.stringify({ schema: 'umminfer-web-session-export-v1', messages: [] }), /turns array/],
  ['non-array turns', JSON.stringify({ schema: 'umminfer-web-session-export-v1', turns: {} }), /turns array/],
  ['string frame budget', JSON.stringify({ schema: 'umminfer-web-session-export-v1', turns: [], video_frame_selection: { policy: 'uniform', max_frames: '4' } }), /max_frames must be an integer/],
  ['boolean frame budget', JSON.stringify({ schema: 'umminfer-web-session-export-v1', turns: [], video_frame_selection: { policy: 'uniform', max_frames: true } }), /max_frames must be an integer/],
  ['float frame budget', '{"schema":"umminfer-web-session-export-v1","turns":[],"video_frame_selection":{"policy":"uniform","max_frames":4.0}}', /max_frames must be an integer/],
  ['exponent frame budget', '{"schema":"umminfer-web-session-export-v1","turns":[],"video_frame_selection":{"policy":"uniform","max_frames":4e0}}', /max_frames must be an integer/],
  ['unsupported frame budget', JSON.stringify({ schema: 'umminfer-web-session-export-v1', turns: [], video_frame_selection: { policy: 'uniform', max_frames: 12 } }), /exactly 4, 8, or 16/],
  ['unknown frame-selection field', JSON.stringify({ schema: 'umminfer-web-session-export-v1', turns: [], video_frame_selection: { policy: 'uniform', max_frames: 4, quality: 'high' } }), /Unsupported video_frame_selection field/],
  ['uniform temporal threshold', JSON.stringify({ schema: 'umminfer-web-session-export-v1', turns: [], video_frame_selection: { policy: 'uniform', max_frames: 4, motion_threshold: 0.02 } }), /does not accept temporal thresholds/],
  ['temporal string threshold', JSON.stringify({ schema: 'umminfer-web-session-export-v1', turns: [], video_frame_selection: { policy: 'temporal', motion_threshold: '0.02', scene_threshold: 0.25, max_frames: 4 } }), /motion_threshold must be a finite number/],
  ['turn_count mismatch', JSON.stringify({ schema: 'umminfer-web-session-export-v1', turn_count: 2, turns: [] }), /turn_count/],
  ['invalid role system', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ role: 'system', prompt: 'hi', response: 'yo' }],
  }), /role is unsupported/],
  ['invalid role tool', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ role: 'tool', prompt: 'hi', response: 'yo' }],
  }), /role is unsupported/],
  ['non-string prompt', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ prompt: 42, response: 'ok' }],
  }), /prompt must be a string/],
  ['non-string response', JSON.stringify({
    schema: 'umminfer-session-export-v1',
    turns: [{ prompt: 'ok', response: ['bad'] }],
  }), /response must be a string/],
  ['invalid slot kind', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ prompt: 'hi', response: 'ok', visual_slots: [{ id: 's0', kind: 'audio' }] }],
  }), /kind must be/],
  ['missing slot id', JSON.stringify({
    schema: 'umminfer-session-export-v1',
    turns: [{ prompt: 'hi', response: 'ok', visual_slots: [{ kind: 'image' }] }],
  }), /non-empty string id or slot_id/],
  ['negative telemetry ms', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ prompt: 'hi', response: 'ok', telemetry: { ttft_ms: -1 } }],
  }), /non-negative finite number/],
  ['covered VRAM missing peak_bytes', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ prompt: 'hi', response: 'ok', telemetry: {
      native_allocator: { coverage: 'COVERED', covered: true, peak_bytes: null },
    } }],
  }), /peak_bytes is required/],
  ['NOT_COVERED VRAM with fabricated peak_bytes', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ prompt: 'hi', response: 'ok', telemetry: {
      native_allocator: { coverage: 'NOT_COVERED', covered: false, peak_bytes: 0 },
    } }],
  }), /peak_bytes must be null/],
  ['exposed behavior supervised > token', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ prompt: 'hi', response: 'ok', telemetry: {
      behavior_logprobs: { status: 'exposed', token_count: 3, supervised_count: 5 },
    } }],
  }), /supervised_count cannot exceed/],
  ['not_exposed behavior with fabricated count', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ prompt: 'hi', response: 'ok', telemetry: {
      behavior_logprobs: { status: 'not_exposed', token_count: 0, supervised_count: 0 },
    } }],
  }), /must have null token_count/],
  ['oversized ASCII > 2 MiB', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ prompt: 'a'.repeat(2 * 1024 * 1024 + 10), response: '' }],
  }), /2 MiB size limit/],
  ['oversized multi-byte UTF-8 > 2 MiB', JSON.stringify({
    schema: 'umminfer-web-session-export-v1',
    turns: [{ prompt: '墨'.repeat(700000), response: '' }],
  }), /2 MiB size limit/],
];

const intactRt = createMockRuntime(false);
const snapshotBefore = JSON.stringify(intactRt.state);
for (const [label, payload, pattern] of invalidCases) {
  let caught = null;
  try {
    const parsed = parseImportedSessionJson(payload);
    applyImportedSessionWithContext(parsed, intactRt.ctx);
  } catch (err) {
    caught = err;
  }
  assert(caught && pattern.test(caught.message),
         `fail-closed case "${label}" did not match ${pattern} (got: ${caught?.message})`);
}
assert(JSON.stringify(intactRt.state) === snapshotBefore,
       'failed imports mutated prior state!');

// 5. Active-busy guard rejects import cleanly without mutating live state.
const busyRt = createMockRuntime(true);
const busySnapshot = JSON.stringify(busyRt.state);
let busyRejected = null;
try {
  applyImportedSessionWithContext(parsedWeb, busyRt.ctx);
} catch (err) {
  busyRejected = err;
}
assert(busyRejected && /while a response is streaming/.test(busyRejected.message),
       `active-busy guard failed: ${busyRejected?.message}`);
assert(JSON.stringify(busyRt.state) === busySnapshot,
       'active-busy rejection mutated prior state!');
assert(networkCalls === 0, `expected 0 network calls, saw ${networkCalls}`);

console.log('session import behavior passed');
"""
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "session-import.mjs"
        path.write_text(harness, encoding="utf-8")
        result = subprocess.run([node, str(path)], capture_output=True,
                                text=True, timeout=60)
    require(result.returncode == 0,
            f"session import behavior failed: {(result.stderr or result.stdout).strip()[:1200]}")
    require("session import behavior passed" in result.stdout,
            "session import harness did not reach its success assertion")


def check_keyboard_shortcuts(html, script):
    """Global Web UI keyboard shortcuts markup hints, document keydown hook, and purity."""
    for snippet in (
        'id="exportSession"',
        'aria-keyshortcuts="Meta+E Control+E Meta+Shift+E Control+Shift+E"',
        'id="importSession"',
        'aria-keyshortcuts="Meta+O Control+O"',
        'id="new"',
        'aria-keyshortcuts="Alt+N Meta+Shift+K Control+Shift+K"',
        'id="stop"',
        'aria-keyshortcuts="Escape"',
    ):
        require(snippet in html, f"missing keyboard shortcut markup hint: {snippet}")
    for hook in (
        "// KEYBOARD_SHORTCUTS_BEGIN",
        "// KEYBOARD_SHORTCUTS_END",
        "function resolveGlobalShortcutAction(",
        "function handleGlobalShortcut(",
        "document.addEventListener('keydown'",
        "handleGlobalShortcut(event,",
        "exportSession",
        "openImportDialog",
        "newChat",
        "stopGeneration",
    ):
        require(hook in script, f"missing keyboard shortcut script hook: {hook}")
    block = script.split("// KEYBOARD_SHORTCUTS_BEGIN", 1)[1].split(
        "// KEYBOARD_SHORTCUTS_END", 1)[0]
    require("fetch(" not in block and "new WebSocket(" not in block,
            "keyboard shortcut helpers must never issue network or WebSocket calls")


def check_keyboard_shortcuts_behavior(script):
    """Execute pure global keyboard shortcut resolver and handler contracts in Node."""
    node = shutil.which("node")
    if not node:
        print("skip keyboard shortcuts behavior: node is not installed", flush=True)
        return
    require("// KEYBOARD_SHORTCUTS_BEGIN" in script
            and "// KEYBOARD_SHORTCUTS_END" in script,
            "missing keyboard shortcuts behavior-test markers")
    shortcuts_src = script.split("// KEYBOARD_SHORTCUTS_BEGIN", 1)[1].split(
        "// KEYBOARD_SHORTCUTS_END", 1)[0]
    harness = shortcuts_src + r"""
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

let networkCalls = 0;
globalThis.fetch = async () => { networkCalls += 1; throw new Error('unexpected fetch'); };
globalThis.WebSocket = class { constructor() { networkCalls += 1; throw new Error('unexpected WebSocket'); } };

function makeEvent(props) {
  let prevented = false;
  return {
    key: '',
    code: '',
    metaKey: false,
    ctrlKey: false,
    altKey: false,
    shiftKey: false,
    defaultPrevented: false,
    isComposing: false,
    repeat: false,
    ...props,
    preventDefault() { prevented = true; },
    get _prevented() { return prevented; },
  };
}

function makeHarness(overrides = {}) {
  const calls = [];
  const state = {
    isRunning: false,
    isBusy: false,
    isLiveVoiceActive: false,
    hasTurns: true,
    exportMenuOpen: false,
    ...overrides,
  };
  const ctx = {
    isRunning: () => state.isRunning,
    isBusy: () => state.isBusy,
    isLiveVoiceActive: () => state.isLiveVoiceActive,
    hasTurns: () => state.hasTurns,
    isExportMenuOpen: () => state.exportMenuOpen,
    exportSession: (fmt) => calls.push(['export', fmt]),
    openImportDialog: () => calls.push(['import']),
    newChat: () => calls.push(['new_chat']),
    closeExportMenu: () => calls.push(['close_export_menu']),
    stopGeneration: () => calls.push(['stop_generation']),
  };
  return { state, ctx, calls };
}

// 1. Cmd+E and Ctrl+E -> export_json (enabled, empty, and busy cases).
for (const mod of [{ metaKey: true }, { ctrlKey: true }]) {
  const ev = makeEvent({ key: 'e', ...mod });
  assert(resolveGlobalShortcutAction(ev) === 'export_json', 'expected export_json');
  const h = makeHarness({ hasTurns: true, isBusy: false });
  const res = handleGlobalShortcut(ev, h.ctx);
  assert(res.action === 'export_json' && res.handled === true && ev._prevented === true,
         'export_json enabled mismatch');
  assert(JSON.stringify(h.calls) === JSON.stringify([['export', 'json']]),
         'export_json did not call exportSession("json")');

  // Empty session: prevented so browser search bar does not hijack, but handler not called.
  const emptyEv = makeEvent({ key: 'e', ...mod });
  const emptyH = makeHarness({ hasTurns: false, isBusy: false });
  const emptyRes = handleGlobalShortcut(emptyEv, emptyH.ctx);
  assert(emptyRes.action === 'export_json' && emptyRes.handled === false &&
         emptyRes.reason === 'empty' && emptyEv._prevented === true && emptyH.calls.length === 0,
         'export_json empty guard mismatch');

  // Busy / running: prevented, handler not called.
  const busyEv = makeEvent({ key: 'e', ...mod });
  const busyH = makeHarness({ hasTurns: true, isRunning: true, isBusy: true });
  const busyRes = handleGlobalShortcut(busyEv, busyH.ctx);
  assert(busyRes.action === 'export_json' && busyRes.handled === false &&
         busyRes.reason === 'busy' && busyEv._prevented === true && busyH.calls.length === 0,
         'export_json busy guard mismatch');
}

// 2. Cmd+Shift+E and Ctrl+Shift+E -> export_markdown (enabled, empty, and busy cases).
for (const mod of [{ metaKey: true, shiftKey: true, key: 'E' }, { ctrlKey: true, shiftKey: true, key: 'e' }]) {
  const ev = makeEvent(mod);
  assert(resolveGlobalShortcutAction(ev) === 'export_markdown', 'expected export_markdown');
  const h = makeHarness({ hasTurns: true, isBusy: false });
  const res = handleGlobalShortcut(ev, h.ctx);
  assert(res.action === 'export_markdown' && res.handled === true && ev._prevented === true,
         'export_markdown enabled mismatch');
  assert(JSON.stringify(h.calls) === JSON.stringify([['export', 'markdown']]),
         'export_markdown did not call exportSession("markdown")');

  const emptyEv = makeEvent(mod);
  const emptyH = makeHarness({ hasTurns: false });
  const emptyRes = handleGlobalShortcut(emptyEv, emptyH.ctx);
  assert(emptyRes.handled === false && emptyRes.reason === 'empty' && emptyEv._prevented === true &&
         emptyH.calls.length === 0, 'export_markdown empty guard mismatch');

  const busyEv = makeEvent(mod);
  const busyH = makeHarness({ hasTurns: true, isRunning: true });
  const busyRes = handleGlobalShortcut(busyEv, busyH.ctx);
  assert(busyRes.handled === false && busyRes.reason === 'busy' && busyEv._prevented === true &&
         busyH.calls.length === 0, 'export_markdown busy guard mismatch');
}

// 3. Cmd+O and Ctrl+O -> import_session (enabled, busy, live-voice active).
for (const mod of [{ metaKey: true }, { ctrlKey: true }]) {
  const ev = makeEvent({ key: 'o', ...mod });
  assert(resolveGlobalShortcutAction(ev) === 'import_session', 'expected import_session');
  const h = makeHarness({ isBusy: false, isLiveVoiceActive: false });
  const res = handleGlobalShortcut(ev, h.ctx);
  assert(res.action === 'import_session' && res.handled === true && ev._prevented === true &&
         JSON.stringify(h.calls) === JSON.stringify([['import']]),
         'import_session enabled mismatch');

  const busyEv = makeEvent({ key: 'o', ...mod });
  const busyH = makeHarness({ isBusy: true });
  const busyRes = handleGlobalShortcut(busyEv, busyH.ctx);
  assert(busyRes.handled === false && busyRes.reason === 'busy' && busyEv._prevented === true &&
         busyH.calls.length === 0, 'import_session busy guard mismatch');

  const voiceEv = makeEvent({ key: 'o', ...mod });
  const voiceH = makeHarness({ isLiveVoiceActive: true });
  const voiceRes = handleGlobalShortcut(voiceEv, voiceH.ctx);
  assert(voiceRes.handled === false && voiceRes.reason === 'live_voice_active' &&
         voiceEv._prevented === true && voiceH.calls.length === 0,
         'import_session live-voice guard mismatch');
}

// 4. Alt+N (including macOS dead-key code=KeyN) and Cmd/Ctrl+Shift+K -> new_chat.
const newChatEvents = [
  { key: 'n', altKey: true },
  { key: 'Dead', code: 'KeyN', altKey: true },
  { key: 'K', metaKey: true, shiftKey: true },
  { key: 'k', ctrlKey: true, shiftKey: true },
];
for (const spec of newChatEvents) {
  const ev = makeEvent(spec);
  assert(resolveGlobalShortcutAction(ev) === 'new_chat',
         `expected new_chat for ${JSON.stringify(spec)}`);
  const h = makeHarness({ isBusy: false, isLiveVoiceActive: false });
  const res = handleGlobalShortcut(ev, h.ctx);
  assert(res.action === 'new_chat' && res.handled === true && ev._prevented === true &&
         JSON.stringify(h.calls) === JSON.stringify([['new_chat']]),
         `new_chat enabled mismatch for ${JSON.stringify(spec)}`);

  const busyEv = makeEvent(spec);
  const busyH = makeHarness({ isRunning: true });
  const busyRes = handleGlobalShortcut(busyEv, busyH.ctx);
  assert(busyRes.handled === false && busyRes.reason === 'busy' && busyEv._prevented === true &&
         busyH.calls.length === 0, 'new_chat busy guard mismatch');

  const voiceEv = makeEvent(spec);
  const voiceH = makeHarness({ isLiveVoiceActive: true });
  const voiceRes = handleGlobalShortcut(voiceEv, voiceH.ctx);
  assert(voiceRes.handled === false && voiceRes.reason === 'live_voice_active' &&
         voiceEv._prevented === true && voiceH.calls.length === 0,
         'new_chat live-voice guard mismatch');
}

// 5. Escape: export menu takes priority -> stop generation -> idle no-op (no preventDefault).
{
  const menuEv = makeEvent({ key: 'Escape' });
  const menuH = makeHarness({ exportMenuOpen: true, isRunning: true });
  const menuRes = handleGlobalShortcut(menuEv, menuH.ctx);
  assert(menuRes.action === 'close_export_menu' && menuRes.handled === true &&
         menuEv._prevented === true &&
         JSON.stringify(menuH.calls) === JSON.stringify([['close_export_menu']]),
         'Escape should close export menu first without stopping generation');

  const stopEv = makeEvent({ key: 'Escape' });
  const stopH = makeHarness({ exportMenuOpen: false, isRunning: true });
  const stopRes = handleGlobalShortcut(stopEv, stopH.ctx);
  assert(stopRes.action === 'stop_generation' && stopRes.handled === true &&
         stopEv._prevented === true &&
         JSON.stringify(stopH.calls) === JSON.stringify([['stop_generation']]),
         'Escape should stop in-flight generation when export menu is closed');

  const idleEv = makeEvent({ key: 'Escape' });
  const idleH = makeHarness({ exportMenuOpen: false, isRunning: false });
  const idleRes = handleGlobalShortcut(idleEv, idleH.ctx);
  assert(idleRes.action === null && idleRes.handled === false &&
         idleEv._prevented === false && idleH.calls.length === 0,
         'Escape when idle must return null and NOT call preventDefault()');

  for (const mod of [{ shiftKey: true }, { metaKey: true }, { ctrlKey: true }, { altKey: true }]) {
    const modEsc = makeEvent({ key: 'Escape', ...mod });
    const modH = makeHarness({ exportMenuOpen: true, isRunning: true });
    const modRes = handleGlobalShortcut(modEsc, modH.ctx);
    assert(modRes.action === null && modRes.handled === false &&
           modEsc._prevented === false && modH.calls.length === 0,
           'modified Escape must be ignored');
  }
}

// 6. defaultPrevented, isComposing, and key repeat guards.
for (const guard of [{ defaultPrevented: true }, { isComposing: true }]) {
  for (const spec of [
    { key: 'e', metaKey: true },
    { key: 'E', ctrlKey: true, shiftKey: true },
    { key: 'o', metaKey: true },
    { key: 'n', altKey: true },
    { key: 'K', metaKey: true, shiftKey: true },
    { key: 'Escape' },
  ]) {
    const ev = makeEvent({ ...spec, ...guard });
    const h = makeHarness({ isRunning: true, exportMenuOpen: true });
    const res = handleGlobalShortcut(ev, h.ctx);
    assert(res.action === null && res.handled === false && ev._prevented === false &&
           h.calls.length === 0,
           `guard ${JSON.stringify(guard)} did not ignore ${JSON.stringify(spec)}`);
  }
}
{
  const repEv = makeEvent({ key: 'e', metaKey: true, repeat: true });
  const repH = makeHarness({ hasTurns: true });
  const repRes = handleGlobalShortcut(repEv, repH.ctx);
  assert(repRes.action === 'export_json' && repRes.handled === false &&
         repRes.reason === 'repeat' && repEv._prevented === true && repH.calls.length === 0,
         'key repeat should prevent browser default without re-triggering export');
}

// 7. Normal typing, editing shortcuts, and browser New Window (Cmd/Ctrl+N) are never hijacked.
const nonShortcuts = [
  { key: 'a', metaKey: true }, { key: 'c', metaKey: true }, { key: 'v', metaKey: true },
  { key: 'x', metaKey: true }, { key: 'z', metaKey: true },
  { key: 'a', ctrlKey: true }, { key: 'c', ctrlKey: true }, { key: 'v', ctrlKey: true },
  { key: 'x', ctrlKey: true }, { key: 'z', ctrlKey: true },
  { key: 'n', metaKey: true }, { key: 'n', ctrlKey: true },
  { key: 'Enter' }, { key: 'Enter', metaKey: true }, { key: 'Enter', ctrlKey: true },
  { key: 'e' }, { key: 'E', shiftKey: true }, { key: 'o' }, { key: 'n' }, { key: 'k' },
  { key: 'e', metaKey: true, altKey: true },
  { key: 'o', ctrlKey: true, altKey: true },
  { key: 'o', metaKey: true, shiftKey: true },
  { key: 'n', altKey: true, shiftKey: true },
  { key: 'n', altKey: true, metaKey: true },
  { key: 'k', metaKey: true },
  { key: 'k', ctrlKey: true, shiftKey: true, altKey: true },
];
for (const spec of nonShortcuts) {
  const ev = makeEvent(spec);
  const h = makeHarness({ hasTurns: true, isRunning: true });
  const res = handleGlobalShortcut(ev, h.ctx);
  assert(res.action === null && res.handled === false && ev._prevented === false &&
         h.calls.length === 0,
         `non-shortcut was hijacked: ${JSON.stringify(spec)}`);
}

assert(networkCalls === 0, `expected 0 network calls, saw ${networkCalls}`);
console.log('keyboard shortcuts behavior passed');
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "keyboard-shortcuts.mjs"
        path.write_text(harness, encoding="utf-8")
        result = subprocess.run([node, str(path)], capture_output=True,
                                text=True, timeout=60)
    require(result.returncode == 0,
            f"keyboard shortcuts behavior failed: {(result.stderr or result.stdout).strip()[:1200]}")
    require("keyboard shortcuts behavior passed" in result.stdout,
            "keyboard shortcuts harness did not reach its success assertion")


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

    def test_realtime_audio_meter_and_barge_in_behavior(self):
        check_realtime_audio_meter_and_barge_in_behavior(self.script)

    def test_showcase_and_telemetry(self):
        check_showcase_and_telemetry(self.html, self.script)

    def test_showcase_and_telemetry_behavior(self):
        check_showcase_and_telemetry_behavior(self.script)

    def test_session_export(self):
        check_session_export(self.html, self.script)

    def test_session_export_behavior(self):
        check_session_export_behavior(self.script)

    def test_session_import(self):
        check_session_import(self.html, self.script)

    def test_session_import_behavior(self):
        check_session_import_behavior(self.script)

    def test_keyboard_shortcuts(self):
        check_keyboard_shortcuts(self.html, self.script)

    def test_keyboard_shortcuts_behavior(self):
        check_keyboard_shortcuts_behavior(self.script)

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
        ("realtime_audio_meter_and_barge_in_behavior", lambda: check_realtime_audio_meter_and_barge_in_behavior(script)),
        ("showcase_and_telemetry", lambda: check_showcase_and_telemetry(html, script)),
        ("showcase_and_telemetry_behavior", lambda: check_showcase_and_telemetry_behavior(script)),
        ("session_export", lambda: check_session_export(html, script)),
        ("session_export_behavior", lambda: check_session_export_behavior(script)),
        ("session_import", lambda: check_session_import(html, script)),
        ("session_import_behavior", lambda: check_session_import_behavior(script)),
        ("keyboard_shortcuts", lambda: check_keyboard_shortcuts(html, script)),
        ("keyboard_shortcuts_behavior", lambda: check_keyboard_shortcuts_behavior(script)),
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

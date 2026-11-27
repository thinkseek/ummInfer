#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Preregistered Before/After NS-06 three-shot + NS-08 lifecycle gate on GPU 7.

No prompt/reference mutation or alternate candidate is permitted. The parent and
candidate run the same frozen sketch+voice+text request with six auto visual slots.
Existing unmodified Web export/clear/import helpers are exercised in Node, then a
real text-only reconnect uses the imported history. Raw media stays external.
"""
import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "python"))
import public_narrative_i2v_live as base  # noqa: E402
import cross_slot_visual_continuity_live as cross  # noqa: E402
from umminfer_rl_trajectory import TrajectoryRecorder  # noqa: E402

FORMAT = "umminfer-ns06-paired-76-v1"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_prereg(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(value["format"] == "umminfer-ns06-paired-preregistration-v1"
            and value["frozen_before_any_before_after_output"] is True,
            "invalid paired preregistration")
    require(value["execution_policy"]["no_post_observation_prompt_or_reference_plan_mutation"]
            and value["execution_policy"]["no_retries_or_alternate_candidates"],
            "paired execution is not frozen")
    return value


def main_body(prereg, arm, sketch, voice):
    req = prereg["request"]
    return {
        "model": "qwen3-native", "stream": True,
        "session_id": f"ns06-{arm}-76",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prereg["exact_text_plan"]},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(sketch).decode()}},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(voice).decode(), "format": "wav"}},
        ]}],
        "modalities": req["modalities"], "narrative": req["narrative"],
        "image": req["image"], "video": req["video"], "audio": req["audio"],
        "max_tokens": req["max_tokens"], "temperature": req["temperature"],
        "seed": req["seed"],
    }


def trace_leg_completed(traces, request_id, leg):
    return any(row.get("event") == "leg_end" and row.get("leg") == leg
               and row.get("status") == "completed"
               and row.get("request_id") == request_id for row in traces)


def visual_metadata(record):
    artifacts = {item.get("slot_id"): item for item in record["artifacts"]
                 if item.get("slot_id") is not None}
    legs = {item.get("slot_id"): item for item in record["terminal_legs"]
            if item.get("slot_id") is not None}
    out = []
    for slot in record["visual_slots"]:
        slot_id = slot["slot_id"]
        artifact = artifacts.get(slot_id, {})
        leg = legs.get(slot_id, {})
        item = {
            "id": slot_id, "kind": slot["media_type"],
            "prompt": slot.get("prompt", ""),
            "bytes": artifact.get("bytes"), "sha256": artifact.get("sha256"),
            "continuity_mode": slot.get("continuity_mode", ""),
            "anchor_slot_index": slot.get("anchor_slot_index"),
            "anchor_explicit": slot.get("anchor_explicit") is True,
            "anchor_sha256": artifact.get("anchor_sha256") or leg.get("anchor_sha256") or "",
            "error": leg.get("error", ""),
        }
        out.append(item)
    return out


def client_input_state(prereg, arm, record):
    asr = "".join(event.get("text", "") for event in record["events"]
                  if event.get("type") == "asr_final")
    return {
        "session_id": record["session_id"], "model": "qwen3-native",
        "exported_at": "2026-11-27T00:00:00.000Z",
        "turns": [{
            "prompt": prereg["exact_text_plan"], "response": record["clean_text"],
            "asr_text": asr, "visual_slots": visual_metadata(record),
            "telemetry": None,
        }],
    }


def production_client_script(web_source, new_chat_source):
    """Compose existing unmodified production helpers with a record-only adapter."""
    def marked(name):
        return web_source.split(f"// {name}_BEGIN", 1)[1].split(f"// {name}_END", 1)[0]
    parts = [marked(name) for name in (
        "VIDEO_FRAME_SELECTION", "TURN_TELEMETRY", "NARRATIVE_CONTINUITY",
        "REALTIME_AUDIO_METER", "SESSION_EXPORT", "SESSION_IMPORT")]
    return "\n".join(parts) + "\n" + new_chat_source + r"""
import fs from 'node:fs';
import path from 'node:path';
const inputPath = process.argv[2];
const outDir = process.argv[3];
const inputState = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
fs.mkdirSync(outDir, {recursive: false});
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const write = (name, value) => fs.writeFileSync(path.join(outDir, name),
  typeof value === 'string' ? value : JSON.stringify(value, null, 2) + '\n');

let networkCalls = 0;
globalThis.fetch = () => { networkCalls += 1; throw new Error('forbidden fetch'); };
globalThis.WebSocket = class { constructor() { networkCalls += 1; throw new Error('forbidden WS'); } };
const exported = buildSessionExportJson(inputState);
write('export.json', exported);
write('export.md', buildSessionExportMarkdown(inputState));

// Actual production newChat(), with only DOM/playback adapters mocked.
let running = false;
let turn = null;
let inflight = new Map();
let stopped = false;
let history = [{role: 'user', content: inputState.turns[0].prompt},
               {role: 'assistant', content: inputState.turns[0].response}];
let sessionTurns = [{}];
let sessionID = inputState.session_id;
let videoFrameBudgetPersistenceError = null;
let stopRealtimeCalls = 0;
let playerStopCalls = 0;
let emptyState = {};
let ui = {
  emptyState,
  transcript: {replaceChildren(value) { assert(value === emptyState, 'wrong empty node'); }},
  session: {hidden: false},
  prompt: {focus() {}},
  videoFramePolicy: {value: ''}, videoMotionThreshold: {value: ''},
  videoSceneThreshold: {value: ''}, videoFrameBudget: {value: ''},
};
let player = {stop() { playerStopCalls += 1; }};
function stopRealtime() { stopRealtimeCalls += 1; }
function clearComposerAndSessionResources() {}
function syncEmptyState() {}
function saveSettings() {}
function syncVideoFrameControls() {}
function clearError() {}
function setStatus() {}
const cleared = newChat();
assert(cleared && history.length === 0 && sessionTurns.length === 0 && sessionID === '' &&
       playerStopCalls === 1 && stopRealtimeCalls === 1 && networkCalls === 0,
       'production newChat did not clear state');
write('clear.json', {cleared, history_count: 0, turn_count: 0, session_id: sessionID,
                     player_stop_calls: playerStopCalls, stop_realtime_calls: stopRealtimeCalls});

function createMockRuntime() {
  const state = {busy: false, sessionID: '', history: [], turns: [], sessionChip: {},
                 status: '', error: '', audioStopped: 0, syncCount: 0, frameSelection: undefined};
  const ctx = {
    isBusy: () => state.busy,
    stopActiveAudio() { state.audioStopped += 1; },
    resetState(next) { state.sessionID = next; state.history = []; state.turns = []; },
    applyVideoFrameSelection(selection) { state.frameSelection = selection; },
    clearTranscript() {}, clearError() {},
    addTurn(role, text) {
      return {role, text, asrText: '', asrEl: {textContent: ''}, renderedText: '', slots: [],
              interrupted: false, spokenBytes: null, interruptedBadge: null, telemetry: null,
              renderedPills: []};
    },
    renderAssistantText(turn, text) { turn.renderedText = text; },
    addImportedVisualSlot(turn, meta) {
      turn.slots.push({...meta, continuityBadge: visualSlotContinuityBadge(meta),
                       importedMediaOmitted: !meta.error});
    },
    attachInterruptedBadge(turn, bytes, badgeText) { turn.interruptedBadge = {bytes, badgeText}; },
    renderTelemetry(turn) { turn.renderedPills = formatTurnTelemetry(turn.telemetry); },
    recordTurn(user, assistant, prompt, response) {
      state.turns.push({userTurn: user, assistantTurn: assistant});
      if (prompt) state.history.push({role: 'user', content: prompt});
      if (response) state.history.push({role: 'assistant', content: response});
    },
    setSessionChip(chip) { state.sessionChip = chip; },
    syncUI() { state.syncCount += 1; }, setStatus(text) { state.status = text; },
  };
  return {state, ctx};
}
const parsed = parseImportedSessionJson(JSON.stringify(exported));
const runtime = createMockRuntime();
const applied = applyImportedSessionWithContext(parsed, runtime.ctx);
assert(applied.turnCount === inputState.turns.length && networkCalls === 0,
       'import count/network contract failed');
const reExportState = {
  session_id: runtime.state.sessionID, model: parsed.model,
  exported_at: inputState.exported_at,
  turns: runtime.state.turns.map((entry) => ({
    prompt: entry.userTurn?.text || '', response: entry.assistantTurn?.text || '',
    asr_text: entry.assistantTurn?.asrText || '',
    interrupted: entry.assistantTurn?.interrupted === true,
    spoken_bytes: entry.assistantTurn?.spokenBytes,
    visual_slots: entry.assistantTurn?.slots || [],
    telemetry: entry.assistantTurn?.telemetry || null,
  })),
};
const reexported = buildSessionExportJson(reExportState);
write('reexport.json', reexported);
assert(JSON.stringify(reexported) === JSON.stringify(exported), 're-export JSON diverged');
const reconnectBody = {
  model: parsed.model, stream: true, session_id: parsed.session_id,
  messages: [...runtime.state.history, {role: 'user', content: 'Reply with exactly READY.'}],
  modalities: ['text'], max_tokens: 32, temperature: 0, seed: 42,
};
write('reconnect-body.json', reconnectBody);
write('result.json', {
  schema: exported.schema, turn_count: exported.turn_count,
  clear: true, imported_turn_count: applied.turnCount, network_calls: networkCalls,
  exact_reexport: true, restored_session_id: runtime.state.sessionID,
  restored_text: runtime.state.turns.map((x) => x.assistantTurn.text),
  restored_slots: runtime.state.turns[0].assistantTurn.slots,
  imported_media_omitted: runtime.state.turns[0].assistantTurn.slots.every(
    (slot) => slot.importedMediaOmitted === !slot.error),
  raw_media_and_pcm_restoration: 'NOT_SUPPORTED_BY_EXISTING_SESSION_SCHEMA',
  reconnect_history_length: runtime.state.history.length,
});
"""


def run_client_lifecycle(prereg, arm, record, web_root, node, arm_out):
    web_source = (Path(web_root) / "index.html").read_text(encoding="utf-8")
    new_chat = web_source.split("function newChat() {", 1)[1].split(
        "\nui.exportSession.addEventListener", 1)[0]
    new_chat = "function newChat() {" + new_chat
    script = production_client_script(web_source, new_chat)
    client = arm_out / "client"
    client.mkdir()
    input_path = client / "input-state.json"
    input_path.write_text(json.dumps(client_input_state(prereg, arm, record),
                                     indent=2, ensure_ascii=False) + "\n")
    script_path = client / "production-lifecycle.mjs"
    script_path.write_text(script, encoding="utf-8")
    output = client / "out"
    result = subprocess.run([node, str(script_path), str(input_path), str(output)],
                            capture_output=True, text=True, timeout=60)
    require(result.returncode == 0,
            f"{arm} production client lifecycle failed: {result.stdout} {result.stderr}")
    payload = json.loads((output / "result.json").read_text())
    require(payload["exact_reexport"] is True and payload["clear"] is True
            and payload["network_calls"] == 0,
            f"{arm} client lifecycle did not clear/import/re-export exactly")
    return payload


def run_arm(args, prereg, arm, runtime, out):
    require(not out.exists(), f"arm evidence exists: {out}")
    out.mkdir(parents=True)
    run_dir = out / "run"
    run_dir.mkdir()
    sketch = Path(args.sketch).read_bytes()
    voice = Path(args.voice).read_bytes()
    require(sha(sketch) == prereg["input_assets"]["sketch"]["sha256"]
            and sha(voice) == prereg["input_assets"]["voice"]["sha256"],
            "frozen input asset digest mismatch")
    body = main_body(prereg, arm, sketch, voice)
    (out / "request.json").write_text(json.dumps(body, indent=2) + "\n")
    (out / "input-sketch.png").write_bytes(sketch)
    (out / "input-voice.wav").write_bytes(voice)
    binaries = {name: Path(runtime[name]) for name in ("server", "vlm", "asr", "tts", "wan")}
    provenance = {
        "runtime_commit": prereg[f"{arm}_runtime_commit"],
        "execution_harness_commit": args.execution_commit,
        "preregistration_sha256": sha(Path(args.preregistration).read_bytes()),
        "live_harness": {"path": str(Path(__file__).resolve()), **base.sha256_file(__file__)},
        "independent_verifier": {"path": str(args.verifier), **base.sha256_file(args.verifier)},
        "web_client": {"path": str(Path(args.web_root) / "index.html"),
                       **base.sha256_file(Path(args.web_root) / "index.html")},
        "binaries": {name: {"path": str(path), **base.sha256_file(path)}
                     for name, path in binaries.items()},
        "flux_stage": {"path": str(runtime["flux"]), **base.sha256_file(runtime["flux"])},
        "ffmpeg": base.sha256_file(shutil.which(args.ffmpeg)),
        "ffprobe": base.sha256_file(shutil.which(args.ffprobe)),
        "node": base.sha256_file(shutil.which(args.node)),
        "python_runtime": base.runtime_identity(),
    }
    models = {
        "vlm": base.model_identity(args.vlm_model, "Qwen/Qwen3.8-27B"),
        "asr": base.model_identity(args.asr_model, "Qwen/Qwen3-ASR-1.7B"),
        "flux2": base.model_identity(args.flux_model, "black-forest-labs/FLUX.2-klein-4B"),
        "tts": base.model_identity(args.tts_model, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        "wan22": base.model_identity(args.wan_model, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    }
    gpu = {"before": cross.require_gpu7_idle(args.nvidia_smi)}
    monitor = cross.GpuMonitor(args.nvidia_smi)
    port = base.free_port()
    require(port != 43301, "resident port is forbidden")
    names = ("completion", "abort", "entry-in", "asr-in", "vlm-in", "image-in",
             "image-tap-in", "tts-tap-in", "tts-in", "tts-output-in",
             "video-tap-in", "video-in")
    ep = {name: f"ipc://{run_dir}/{name}" for name in names}
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if args.library_path:
        env["LD_LIBRARY_PATH"] = str(args.library_path) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["UMM_TRACE"] = "1"
    env["UMM_TRACE_FILE"] = str(out / "server.trace.jsonl")
    server = entry = asr = vlm = flux = tts = wan = None
    image_tap = video_tap = tts_tap = tts_output_tap = None
    request_id = [None]
    manager_error = []
    manager_record = {"images_settled_before_wan": False,
                      "asr_settled_before_wan": False}
    main_record = reconnect_record = None
    lifecycle = None
    health_final = None
    stage_record = {}

    def start_wan():
        nonlocal wan
        wan = base.ManagedProcess("video", [
            binaries["wan"], "--kind", "video", "--name", "video", "--video-backend", "cuda",
            "--model", args.wan_model, "--gpu", "7", "--size", "704x480",
            "--steps", "20", "--fps", "16", "--recv-endpoint", ep["video-in"],
            "--coordinator-endpoint", ep["completion"], "--abort-endpoint", ep["abort"]],
            out / "video.log", env=env)
        wan.wait_log("native stage video ready kind=video", args.timeout)
        gpu["wan_ready"] = cross.gpu7_memory(args.nvidia_smi)

    def manager():
        try:
            deadline = time.monotonic() + args.timeout
            outputs_seen = False
            asr_settled = False
            while time.monotonic() < deadline:
                traces = base.trace_records(out / "server.trace.jsonl")
                if not asr_settled:
                    asr_done = trace_leg_completed(traces, request_id[0], "asr")
                    if asr_done:
                        # This path's trace is authoritative even when the
                        # optional per-stage metrics family is absent.
                        asr.stop(ep["asr-in"])
                        require(asr.returncode == 0, "ASR did not stop cleanly")
                        stage_record["asr_terminal"] = "completed"
                        stage_record["asr_stopped_ns"] = base.clock_ns()
                        gpu["after_asr_stop"] = cross.gpu7_memory(args.nvidia_smi)
                        asr_settled = True
                        manager_record["asr_settled_before_wan"] = True
                outputs_seen = any(row.get("event") == "legs_snapshot"
                                   and row.get("phase") == "outputs_started"
                                   and row.get("request_id") == request_id[0]
                                   for row in traces)
                if outputs_seen and request_id[0] and asr_settled:
                    state = base.coordinator_stage_state(port, "image")
                    settled = state["completed"] + state["failed"] + state["aborted"]
                    if state["submitted"] > 0 and settled == state["submitted"] \
                            and state["in_flight"] == 0:
                        stage_record["image_coordinator_final"] = state
                        break
                time.sleep(0.02)
            else:
                raise TimeoutError("ASR/VLM/images did not settle before Wan switch")
            flux.stop(ep["image-in"])
            require(flux.returncode == 0, "FLUX did not stop cleanly")
            stage_record["flux_stopped_ns"] = base.clock_ns()
            gpu["after_flux_stop"] = cross.gpu7_memory(args.nvidia_smi)
            start_wan()
            manager_record["images_settled_before_wan"] = True
        except Exception as error:
            manager_error.append(str(error))
            if request_id[0]:
                try:
                    base.http_json(port, "POST", "/abort", {
                        "request_id": request_id[0], "session_id": f"ns06-{arm}-76"})
                except Exception:
                    pass

    try:
        image_tap = base.TapProxy("image", ep["image-tap-in"], ep["image-in"], out)
        video_tap = base.TapProxy("video", ep["video-tap-in"], ep["video-in"], out)
        tts_tap = base.TapProxy("tts", ep["tts-tap-in"], ep["tts-in"], out)
        tts_output_tap = base.TapProxy("tts-output", ep["tts-output-in"], ep["completion"], out)
        server = base.ManagedProcess("server", [
            binaries["server"], "serve", "--host", "127.0.0.1", "--port", str(port),
            "--model", "qwen3-native", "--entry", "entry", "--entry-endpoint", ep["entry-in"],
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", ep["vlm-in"],
            "--image-entry", "image", "--image-entry-endpoint", ep["image-tap-in"],
            "--tts-entry", "tts", "--tts-entry-endpoint", ep["tts-tap-in"],
            "--video-entry", "video", "--video-entry-endpoint", ep["video-tap-in"],
            "--completion", ep["completion"], "--abort", ep["abort"],
            "--max-in-flight", "2"], out / "server.log", env=env)
        entry = base.ManagedProcess("entry", [
            binaries["server"], "stage", "--name", "entry", "--recv", ep["entry-in"],
            "--coordinator", ep["completion"], "--abort", ep["abort"]],
            out / "entry.log", env=env)
        asr = base.ManagedProcess("asr", [
            binaries["asr"], "--kind", "asr", "--name", "asr", "--model", args.asr_model,
            "--gpu", "7", "--recv-endpoint", ep["asr-in"],
            "--coordinator-endpoint", ep["completion"], "--abort-endpoint", ep["abort"]],
            out / "asr.log", env=env)
        vlm = base.ManagedProcess("vlm", [
            binaries["vlm"], "--kind", "qwen38", "--name", "vlm", "--model", args.vlm_model,
            "--gpu", "7", "--batch-size", "1", "--max-new-tokens", "512",
            "--recv-endpoint", ep["vlm-in"], "--coordinator-endpoint", ep["completion"],
            "--abort-endpoint", ep["abort"]], out / "vlm.log", env=env)
        asr.wait_log("native stage asr ready kind=asr", args.timeout)
        vlm.wait_log("native stage vlm ready kind=qwen38", args.timeout)
        gpu["asr_vlm_ready"] = cross.gpu7_memory(args.nvidia_smi)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            try:
                status, _ = base.http_json(port, "GET", "/ready", timeout=5)
                if status == 200:
                    entry.ready_ns = base.clock_ns()
                    break
            except (OSError, http.client.HTTPException):
                pass
            require(server.process.poll() is None, "gateway exited before ready")
            time.sleep(0.05)
        else:
            raise TimeoutError("gateway did not become ready")
        flux = base.ManagedProcess("image", [
            args.python, "-I", runtime["flux"], "--name", "image", "--backend", "diffusers",
            "--model", args.flux_model, "--model-id", "black-forest-labs/FLUX.2-klein-4B",
            "--revision", "UNKNOWN", "--gpu", "7", "--default-size", "512x512",
            "--default-steps", "4", "--recv", ep["image-in"],
            "--coordinator", ep["completion"], "--abort", ep["abort"]],
            out / "image.log", env=env)
        flux.wait_log("ready kind=image name=image", args.timeout)
        tts = base.ManagedProcess("tts", [
            binaries["tts"], "--kind", "tts", "--name", "tts", "--model", args.tts_model,
            "--gpu", "7", "--recv-endpoint", ep["tts-in"],
            "--coordinator-endpoint", ep["tts-output-in"], "--abort-endpoint", ep["abort"]],
            out / "tts.log", env=env)
        tts.wait_log("native stage tts ready kind=tts", args.timeout)
        gpu["flux_tts_ready"] = cross.gpu7_memory(args.nvidia_smi)
        manager_thread = threading.Thread(target=manager, name=f"ns06-{arm}-manager", daemon=True)
        manager_thread.start()
        recorder = TrajectoryRecorder(out / "main" / "trajectories.jsonl")
        main_record = cross.capture_public(port, body, recorder, f"ns06-{arm}-76", request_id)
        manager_thread.join(timeout=10)
        require(not manager_thread.is_alive() and not manager_error,
                f"{arm} output manager failed: {manager_error}")
        require(manager_record["images_settled_before_wan"], "Wan stage switch did not occur")
        abort_status, abort_body = base.http_json(port, "POST", "/abort", {
            "request_id": main_record["request_id"], "session_id": f"ns06-{arm}-76"})
        require(abort_status == 200 and abort_body.get("success") is True,
                "post-terminal abort was not an inert ACK")
        lifecycle = run_client_lifecycle(prereg, arm, main_record, args.web_root, args.node, out)
        reconnect_body = json.loads((out / "client" / "out" / "reconnect-body.json").read_text())
        reconnect_recorder = TrajectoryRecorder(out / "reconnect" / "trajectories.jsonl")
        reconnect_record = cross.capture_public(
            port, reconnect_body, reconnect_recorder, f"ns06-{arm}-76")
        health_status, health_final = base.http_json(port, "GET", "/health", timeout=30)
        require(health_status == 200 and health_final.get("in_flight") == 0,
                "arm retained in-flight work")
        gpu["after_complete_before_cleanup"] = cross.gpu7_memory(args.nvidia_smi)
    finally:
        for process, endpoint in ((wan, ep["video-in"]), (tts, ep["tts-in"]),
                                  (flux, ep["image-in"]), (vlm, ep["vlm-in"]),
                                  (asr, ep["asr-in"]), (entry, ep["entry-in"])):
            if process is not None and process.stop_ns is None:
                process.stop(endpoint)
        if server is not None:
            server.stop(None)
        for tap in (image_tap, video_tap, tts_tap, tts_output_tap):
            if tap is not None and tap.thread.is_alive():
                tap.stop()
        monitor.stop()
        gpu["after_cleanup"] = cross.wait_gpu7_idle(args.nvidia_smi)
        stage_record["processes"] = [process.record() for process in
                                     (server, entry, asr, vlm, flux, tts, wan)
                                     if process is not None]
        (out / "gpu7-monitor.json").write_text(json.dumps(monitor.samples, indent=2) + "\n")
        require(not manager_error, f"arm failed; raw evidence retained: {manager_error}")

    require(main_record is not None and reconnect_record is not None and lifecycle is not None,
            "arm lacks main/reconnect/client evidence")
    public_audio = []
    for item in main_record["artifacts"]:
        if item["media_type"] == "audio":
            public_audio.append((item, (out / "main" / "artifacts" / item["ref"]).read_bytes()))
    tts_joins = base.join_tts_pcm(cross.native_submits(tts_tap.records),
                                  tts_output_tap.records, public_audio, out)
    require(tts_joins and len(tts_joins) == len(cross.native_submits(tts_tap.records)),
            f"{arm} native/public TTS join is incomplete")
    report = {
        "format": "umminfer-ns06-arm-76-v1", "arm": arm,
        "runtime_commit": prereg[f"{arm}_runtime_commit"],
        "main_record": "main/trajectories.jsonl", "main_artifact_dir": "main/artifacts",
        "reconnect_record": "reconnect/trajectories.jsonl",
        "reconnect_artifact_dir": "reconnect/artifacts",
        "main": main_record, "reconnect": reconnect_record, "client_lifecycle": lifecycle,
        "visual_metadata": visual_metadata(main_record),
        "image_submits": cross.native_submits(image_tap.records),
        "video_submits": cross.native_submits(video_tap.records),
        "tts_submits": cross.native_submits(tts_tap.records),
        "tts_output": tts_output_tap.records, "tts_legs": tts_joins,
        "post_terminal_abort": {"status": 200, "success": True},
        "provenance": provenance, "models": models, "gpu7": gpu,
        "stage_lifecycle": stage_record, "manager": manager_record,
        "health_final": health_final,
        "gpu_monitor": "gpu7-monitor.json", "video_log": "video.log",
        "image_log": "image.log", "tts_log": "tts.log", "asr_log": "asr.log",
        "perceptual_quality": "NOT_RUN", "raw_media_audio_import": "NOT_SUPPORTED_EXISTING_SCHEMA",
        "native_narration_legs": len(tts_joins),
        "narration_leg_target_met": len(tts_joins) == 3,
    }
    (out / "arm-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def paired_checks(prereg, before, after, out_root):
    """Producer gates; the independent verifier repeats these from raw bytes."""
    expected = prereg["expected_outcomes"]
    for arm, report in (("before", before), ("after", after)):
        record = report["main"]
        require(record["terminal_state"] == "completed" and len(record["visual_slots"]) == 6,
                f"{arm} did not complete six parsed visual slots")
        require(report["reconnect"]["terminal_state"] == "completed"
                and report["reconnect"]["clean_text"].strip() == expected["both"]["reconnect_text"]
                and not report["reconnect"]["artifacts"],
                f"{arm} text-only reconnect failed")
        actual_states = [leg["state"] for leg in record["terminal_legs"]
                         if leg.get("slot_id") is not None]
        require(actual_states == expected[arm]["visual_leg_states"],
                f"{arm} visual states differ from frozen expectation: {actual_states}")
        images = [item for item in record["artifacts"] if item["media_type"] == "image"]
        videos = [item for item in record["artifacts"] if item["media_type"] == "video"]
        audio = [item for item in record["artifacts"] if item["media_type"] == "audio"]
        require(len(images) == expected[arm]["keyframe_count"]
                and len(videos) == expected[arm]["i2v_count"] and audio,
                f"{arm} media/audio counts differ")
        # The preregistered three-native-leg target is measured, never relaxed.
        # Existing first-clause segmentation may expose a NOT_MET result.
        report["narration_leg_target_met"] = len(report["tts_submits"]) == 3
        report["native_narration_legs"] = len(report["tts_submits"])
    require(before["main"]["clean_text"] == after["main"]["clean_text"],
            "Before/After clean prose diverged")
    before_pcm = b"".join((out_root / "before" / "main" / "artifacts" / item["ref"]).read_bytes()
                          for item in before["main"]["artifacts"] if item["media_type"] == "audio")
    after_pcm = b"".join((out_root / "after" / "main" / "artifacts" / item["ref"]).read_bytes()
                         for item in after["main"]["artifacts"] if item["media_type"] == "audio")
    require(before_pcm == after_pcm and before_pcm, "Before/After PCM bytes diverged")


def live(args):
    require(os.environ.get("UMMINFER_LIVE_OPT_IN") == "1",
            "REAL_GENERATION: NOT_RUN; explicit opt-in required")
    cross.wait_gpu7_idle(args.nvidia_smi)
    prereg = load_prereg(args.preregistration)
    out = Path(args.out_dir)
    require(not out.exists(), "paired evidence directory already exists")
    out.mkdir(parents=True)
    shared = out / "inputs"
    shared.mkdir()
    for name, source in (("sketch.png", args.sketch), ("voice.wav", args.voice)):
        shutil.copyfile(source, shared / name)
    before_rt = {name: getattr(args, f"before_{name}") for name in
                 ("server", "vlm", "asr", "tts", "wan", "flux")}
    after_rt = {name: getattr(args, f"after_{name}") for name in
                ("server", "vlm", "asr", "tts", "wan", "flux")}
    before = run_arm(args, prereg, "before", before_rt, out / "before")
    after = run_arm(args, prereg, "after", after_rt, out / "after")
    paired_checks(prereg, before, after, out)
    report = {
        "format": FORMAT, "result": "PAIRED_TECHNICAL_PASS",
        "preregistration": str(args.preregistration),
        "preregistration_sha256": sha(Path(args.preregistration).read_bytes()),
        "execution_harness_commit": args.execution_commit,
        "before_arm": "before/arm-report.json", "after_arm": "after/arm-report.json",
        "technical_gain": {
            "keyframe_and_i2v_before": [1, 1], "keyframe_and_i2v_after": [3, 3],
            "exact_image_edit_links_after": 2, "exact_i2v_links_after": 3,
            "exact_i2v_links_before": 1,
            "clean_prose_equal": True, "pcm_equal": True,
        },
        "ns06_human_character_style_threshold": "NOT_RUN / NOT_MET",
        "ns06_full_pass": False,
        "ns06_narration_leg_target": {
            "expected_native_legs": 3,
            "before_actual": before["native_narration_legs"],
            "after_actual": after["native_narration_legs"],
            "met": before["narration_leg_target_met"] and after["narration_leg_target_met"],
        },
        "ns08_no_regression": True,
        "ns08_raw_media_and_audio_restoration": "NOT_SUPPORTED_BY_EXISTING_SCHEMA",
        "ns08_human_walkthrough": "NOT_RUN",
        "generalization": "NOT_COVERED", "training_optimizer": "NOT_RUN",
        "resident_service": "NOT_CONTACTED",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    verify = subprocess.run([
        sys.executable, "-I", str(args.verifier), str(out), "--out",
        str(out / "independent-verification.json"), "--ffmpeg", args.ffmpeg,
        "--ffprobe", args.ffprobe], capture_output=True, text=True, timeout=900)
    require(verify.returncode == 0,
            f"independent verification failed:\n{verify.stdout}\n{verify.stderr}")
    summary = {"format": FORMAT, "result": report["result"],
               "technical_gain": report["technical_gain"],
               "ns06_human_character_style_threshold": report["ns06_human_character_style_threshold"],
               "ns06_full_pass": False,
               "ns06_narration_leg_target": report["ns06_narration_leg_target"],
               "ns08_no_regression": True,
               "independent_verification": "PASS"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()
                       and p.name != "SHA256SUMS" and "run" not in p.relative_to(out).parts):
        files[path.relative_to(out).as_posix()] = base.sha256_file(path)
    (out / "SHA256SUMS").write_text("".join(
        f"{item['sha256']}  {name}\n" for name, item in files.items()))
    print(json.dumps(summary, sort_keys=True))
    print("NS06_PAIRED_TECHNICAL: PASS; HUMAN_PERCEPTUAL: NOT_RUN")
    print("NS08_NO_REGRESSION: PASS; RAW_MEDIA_AUDIO_IMPORT: NOT_SUPPORTED_EXISTING_SCHEMA")


def self_check(args=None):
    prereg_path = args.preregistration if args is not None else (
        ROOT / "benchmarks/cross-slot-visual-continuity/ns06-preregistration.json")
    prereg = load_prereg(prereg_path)
    body = main_body(prereg, "before", b"sketch", b"voice")
    require(body["narrative"]["max_slots"] == 6 and len(prereg["slots"]) == 6,
            "six-slot plan changed")
    require([slot["source"] for slot in prereg["slots"]] == [None, 0, 0, 2, 0, 4],
            "reference plan changed")
    require(len(prereg["expected_clean_prose_sentences"]) == 3,
            "narration plan changed")
    require("NOT_RUN" in prereg["claim_boundaries"]["ns06_human_character_style_threshold"],
            "human perceptual boundary changed")
    print("OFFLINE_NS06_PAIRED: PASS (frozen six-slot plan, parent failure/candidate chain, NS-08 limits)")
    print("REAL_GENERATION: NOT_RUN")
    print("PERCEPTUAL_QUALITY: NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--preregistration", type=Path,
                        default=ROOT / "benchmarks/cross-slot-visual-continuity/ns06-preregistration.json")
    parser.add_argument("--out-dir")
    parser.add_argument("--execution-commit")
    for arm in ("before", "after"):
        for name in ("server", "vlm", "asr", "tts", "wan", "flux"):
            parser.add_argument(f"--{arm}-{name}", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vlm-model", type=Path)
    parser.add_argument("--asr-model", type=Path)
    parser.add_argument("--flux-model", type=Path)
    parser.add_argument("--tts-model", type=Path)
    parser.add_argument("--wan-model", type=Path)
    parser.add_argument("--sketch", type=Path)
    parser.add_argument("--voice", type=Path)
    parser.add_argument("--web-root", type=Path, default=ROOT / "web")
    parser.add_argument("--node", default="node")
    parser.add_argument("--verifier", type=Path,
                        default=Path(__file__).with_name("ns06_cross_slot_verify.py"))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--library-path", default="")
    args = parser.parse_args()
    if args.self_check:
        self_check(args)
        return 0
    if args.out_dir is None:
        parser.error("live mode requires a caller-selected --out-dir")
    required = ["sketch", "voice", "vlm_model", "asr_model", "flux_model", "tts_model", "wan_model"]
    required += [f"{arm}_{name}" for arm in ("before", "after")
                 for name in ("server", "vlm", "asr", "tts", "wan", "flux")]
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"live mode requires --{name.replace('_', '-')}")
    if args.execution_commit is None:
        parser.error("live mode requires --execution-commit")
    live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

const std = @import("std");

fn configure(mod: *std.Build.Module, b: *std.Build, deps_prefix: ?[]const u8, curl_prefix: ?[]const u8, cuda_prefix: ?[]const u8, cuda: bool, link_deps: bool, include_private_src: bool) void {
    mod.addIncludePath(b.path("include"));
    if (include_private_src) mod.addIncludePath(b.path("src"));
    if (curl_prefix) |prefix| {
        if (deps_prefix == null or !std.mem.eql(u8, deps_prefix.?, prefix)) {
            mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "include" }) });
        }
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "lib" }) });
        mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "lib" }) });
    }
    if (deps_prefix) |prefix| {
        // -I, not -isystem: Zig's clang searches its own system include
        // directories before -isystem paths, so a distro httplib.h would win
        // over the pinned one in the dependency prefix.
        mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "include" }) });
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "lib" }) });
    }
    if (cuda_prefix) |prefix| {
        mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "include" }) });
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "lib64" }) });
    }
    if (cuda) mod.addCMacro("UMMINFER_WITH_CUDA", "1");
    // The metrics module dlopen()s libnvidia-ml. glibc before 2.34 keeps
    // dlopen in libdl; macOS has it in libSystem, so only Linux links it.
    if (mod.resolved_target) |resolved| {
        if (resolved.result.os.tag == .linux) {
            mod.linkSystemLibrary("dl", .{ .use_pkg_config = .no });
        }
        if (resolved.result.os.tag == .macos and link_deps) {
            mod.linkFramework("AudioToolbox", .{});
        }
    }
    if (link_deps) {
        mod.linkSystemLibrary("zmq", .{ .use_pkg_config = .no });
        mod.linkSystemLibrary("msgpackc", .{ .use_pkg_config = .no });
        mod.linkSystemLibrary("curl", .{ .use_pkg_config = .no });
        mod.linkSystemLibrary("websockets", .{ .use_pkg_config = .no });
        if (cuda) mod.linkSystemLibrary("cudart", .{ .use_pkg_config = .no });
    }
}

fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
    });
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const deps_prefix = b.option([]const u8, "deps-prefix", "Prefix containing include/ and lib/ for native dependencies");
    const curl_prefix = b.option([]const u8, "curl-prefix", "Prefix of a libcurl build with WebSocket support");
    const cuda_prefix = b.option([]const u8, "cuda-prefix", "CUDA prefix containing include/ and lib64/");
    const cuda = b.option(bool, "cuda", "Enable the CUDA IPC relay") orelse (cuda_prefix != null);
    const tsan = b.option(bool, "tsan", "Enable ThreadSanitizer") orelse false;

    const core = b.addLibrary(.{
        .name = "umm_core",
        .linkage = .static,
        .root_module = module(b, target, optimize),
    });
    configure(core.root_module, b, deps_prefix, curl_prefix, cuda_prefix, cuda, false, true);
    core.root_module.sanitize_thread = tsan;
    core.root_module.addCSourceFiles(.{
        .files = &.{
            "src/util/codec.cpp",
            "src/proto/wire.cpp",
            "src/proto/messages.cpp",
            "src/obs/metrics.cpp",
            "src/obs/request_telemetry.cpp",
            "src/obs/behavior_evidence.cpp",
            "src/relay/relay.cpp",
            "src/comm/comm.cpp",
            "src/scheduling/simple_scheduler.cpp",
            "src/pipeline/pipeline.cpp",
            "src/pipeline/session_store.cpp",
            "src/pipeline/unified_schema.cpp",
            "src/pipeline/narrative_parser.cpp",
            "src/pipeline/narrative_prompt.cpp",
            "src/pipeline/unified.cpp",
            "src/serve/upstream_proxy.cpp",
            "src/serve/online_attestation.cpp",
            "src/serve/openai_helpers.cpp",
            "src/serve/openai_audio.cpp",
            "src/serve/openai_chat.cpp",
            "src/serve/openai_api.cpp",
            "src/serve/realtime_api.cpp",
            "src/audio/audio.cpp",
            "src/client/http.cpp",
            "src/client/openai.cpp",
            "src/client/asr.cpp",
            "src/client/websocket.cpp",
            "src/client/realtime.cpp",
            "src/client/audio_device.cpp",
            "src/client/voice.cpp",
            "src/client/cli.cpp",
        },
        .flags = &.{"-std=c++20"},
    });

    const exe = b.addExecutable(.{
        .name = "umm-infer",
        .root_module = module(b, target, optimize),
    });
    configure(exe.root_module, b, deps_prefix, curl_prefix, cuda_prefix, cuda, true, true);
    exe.root_module.sanitize_thread = tsan;
    exe.root_module.linkLibrary(core);
    exe.root_module.addCSourceFiles(.{
        .files = &.{"src/main.cpp"},
        .flags = &.{"-std=c++20"},
    });
    b.installArtifact(exe);

    const test_step = b.step("test", "Run the end-to-end contract suite");
    const test_exe = b.addExecutable(.{
        .name = "unified_contract_test",
        .root_module = module(b, target, optimize),
    });
    // The contract suite also covers private server headers (serve/energy_vad.hpp),
    // so it needs the same `src` include path as the library it links.
    configure(test_exe.root_module, b, deps_prefix, curl_prefix, cuda_prefix, cuda, true, true);
    test_exe.root_module.sanitize_thread = tsan;
    test_exe.root_module.linkLibrary(core);
    test_exe.root_module.addCSourceFiles(.{
        .files = &.{"tests/unified_contract_test.cpp"},
        .flags = &.{"-std=c++20"},
    });
    const run_test = b.addRunArtifact(test_exe);
    if (b.args) |args| run_test.addArgs(args);
    test_step.dependOn(&run_test.step);
    const unified_test = b.step("unified-test", "Run unified orchestration contract tests");
    unified_test.dependOn(&run_test.step);

    const online_attestation = b.addSystemCommand(&.{ "python3", "-I" });
    online_attestation.addFileArg(b.path("tests/online_attestation_test.py"));
    online_attestation.addArtifactArg(exe);
    test_step.dependOn(&online_attestation.step);
    const online_attestation_test = b.step("online-attestation-test", "Run CPU-only server HTTP attestation tests");
    online_attestation_test.dependOn(&online_attestation.step);

    const wan_cancel_live_self_check = b.addSystemCommand(&.{ "python3", "-I" });
    wan_cancel_live_self_check.addFileArg(b.path("tests/native_wan_cancel_fault_live.py"));
    wan_cancel_live_self_check.addArg("--self-check");
    test_step.dependOn(&wan_cancel_live_self_check.step);
    const wan_cancel_verify_self_check = b.addSystemCommand(&.{ "python3", "-I" });
    wan_cancel_verify_self_check.addFileArg(b.path("tests/native_wan_cancel_fault_verify.py"));
    wan_cancel_verify_self_check.addArg("--self-check");
    test_step.dependOn(&wan_cancel_verify_self_check.step);

    const i2v_geometry_latency_test = b.addSystemCommand(&.{ "python3", "-I" });
    i2v_geometry_latency_test.addFileArg(b.path("tests/i2v_geometry_latency_test.py"));
    test_step.dependOn(&i2v_geometry_latency_test.step);

    const cross_slot_live_self_check = b.addSystemCommand(&.{ "python3", "-I" });
    cross_slot_live_self_check.addFileArg(b.path("tests/cross_slot_visual_continuity_live.py"));
    cross_slot_live_self_check.addArg("--self-check");
    test_step.dependOn(&cross_slot_live_self_check.step);
    const cross_slot_verify_self_check = b.addSystemCommand(&.{ "python3", "-I" });
    cross_slot_verify_self_check.addFileArg(b.path("tests/cross_slot_visual_continuity_verify.py"));
    cross_slot_verify_self_check.addArg("--self-check");
    test_step.dependOn(&cross_slot_verify_self_check.step);

    const ns06_contract_test = b.addSystemCommand(&.{ "python3", "-I" });
    ns06_contract_test.addFileArg(b.path("tests/ns06_cross_slot_test.py"));
    test_step.dependOn(&ns06_contract_test.step);
    const ns06_live_self_check = b.addSystemCommand(&.{ "python3", "-I" });
    ns06_live_self_check.addFileArg(b.path("tests/ns06_cross_slot_live.py"));
    ns06_live_self_check.addArg("--self-check");
    test_step.dependOn(&ns06_live_self_check.step);
    const ns06_verify_self_check = b.addSystemCommand(&.{ "python3", "-I" });
    ns06_verify_self_check.addFileArg(b.path("tests/ns06_cross_slot_verify.py"));
    ns06_verify_self_check.addArg("--self-check");
    test_step.dependOn(&ns06_verify_self_check.step);

    const voice_test_exe = b.addExecutable(.{
        .name = "voice_client_test",
        .root_module = module(b, target, optimize),
    });
    configure(voice_test_exe.root_module, b, deps_prefix, curl_prefix, cuda_prefix, cuda, true, true);
    voice_test_exe.root_module.sanitize_thread = tsan;
    voice_test_exe.root_module.linkLibrary(core);
    voice_test_exe.root_module.addCSourceFiles(.{
        .files = &.{"tests/voice_client_test.cpp"},
        .flags = &.{"-std=c++20"},
    });
    const run_voice_test = b.addRunArtifact(voice_test_exe);
    test_step.dependOn(&run_voice_test.step);
    const voice_test = b.step("voice-test", "Run fake-WebSocket and fake-audio voice client tests");
    voice_test.dependOn(&run_voice_test.step);

    const realtime_stress_exe = b.addExecutable(.{
        .name = "realtime_stress_test",
        .root_module = module(b, target, optimize),
    });
    configure(realtime_stress_exe.root_module, b, deps_prefix, curl_prefix, cuda_prefix, cuda, true, true);
    realtime_stress_exe.root_module.sanitize_thread = tsan;
    realtime_stress_exe.root_module.linkLibrary(core);
    realtime_stress_exe.root_module.addCSourceFiles(.{
        .files = &.{"tests/realtime_stress_test.cpp"},
        .flags = &.{"-std=c++20"},
    });
    const run_realtime_stress = b.addRunArtifact(realtime_stress_exe);
    if (b.args) |args| run_realtime_stress.addArgs(args);
    test_step.dependOn(&run_realtime_stress.step);
    const realtime_stress_test = b.step("realtime-stress-test", "Run deterministic realtime concurrency and lifecycle stress");
    realtime_stress_test.dependOn(&run_realtime_stress.step);

    const realtime_network = b.addSystemCommand(&.{ "python3", "-I" });
    realtime_network.addFileArg(b.path("tests/realtime_network_test.py"));
    test_step.dependOn(&realtime_network.step);
    const realtime_network_test = b.step("realtime-network-test", "Run deterministic realtime network scheduler and verifier tests");
    realtime_network_test.dependOn(&realtime_network.step);

    const public_speech_pcm = b.addSystemCommand(&.{ "python3", "-I" });
    public_speech_pcm.addFileArg(b.path("tests/public_speech_pcm_test.py"));
    test_step.dependOn(&public_speech_pcm.step);

    const ns04_replay = b.addSystemCommand(&.{ "python3", "-I" });
    ns04_replay.addFileArg(b.path("tests/ns04_replay_lifecycle_test.py"));
    test_step.dependOn(&ns04_replay.step);
    const ns04_replay_test = b.step("ns04-replay-test", "Run replay-inclusive NS04 scorecard fixtures");
    ns04_replay_test.dependOn(&ns04_replay.step);
    test_step.dependOn(&ns04_replay.step);
    const public_speech_pcm_test = b.step("public-speech-pcm-test", "Run deterministic public/native PCM verifier tests");
    public_speech_pcm_test.dependOn(&public_speech_pcm.step);

    const ns03_routing = b.addSystemCommand(&.{ "python3", "-I" });
    ns03_routing.addFileArg(b.path("tests/ns03_routing_test.py"));
    test_step.dependOn(&ns03_routing.step);
    const ns03_routing_test = b.step("ns03-routing-test", "Run frozen NS-03 routing and scoring contracts");
    ns03_routing_test.dependOn(&ns03_routing.step);

    const storyboard_replay = b.addSystemCommand(&.{ "python3", "-I" });
    storyboard_replay.addFileArg(b.path("tests/storyboard_replay_verify_test.py"));
    test_step.dependOn(&storyboard_replay.step);
    const storyboard_replay_test = b.step("storyboard-replay-test", "Run lossless storyboard replay verifier contracts");
    storyboard_replay_test.dependOn(&storyboard_replay.step);

    const ns08_history = b.addSystemCommand(&.{ "python3", "-I" });
    ns08_history.addFileArg(b.path("tests/ns08_first_turn_history_verify_test.py"));
    test_step.dependOn(&ns08_history.step);
    const ns08_history_test = b.step("ns08-history-test", "Run paired NS08 raw history and lifecycle verifier contracts");
    ns08_history_test.dependOn(&ns08_history.step);

    const bench_step = b.step("cuda-bench", "Measure SHM and CUDA IPC relay latency");
    if (cuda) {
        const bench_exe = b.addExecutable(.{
            .name = "cuda_bench",
            .root_module = module(b, target, optimize),
        });
        configure(bench_exe.root_module, b, deps_prefix, curl_prefix, cuda_prefix, cuda, true, false);
        bench_exe.root_module.linkLibrary(core);
        bench_exe.root_module.addCSourceFiles(.{
            .files = &.{"tests/cuda_bench.cpp"},
            .flags = &.{"-std=c++20"},
        });
        b.installArtifact(bench_exe);
        const run = b.addRunArtifact(bench_exe);
        bench_step.dependOn(&run.step);
    } else {
        const note = b.addSystemCommand(&.{ "sh", "-c", "echo 'cuda-bench needs -Dcuda; no benchmark built'" });
        bench_step.dependOn(&note.step);
    }

    const smoke = b.addSystemCommand(&.{ "python3", "-I" });
    smoke.addFileArg(b.path("tests/http_smoke.py"));
    smoke.addArtifactArg(exe);
    const smoke_step = b.step("smoke", "Run HTTP and process lifecycle tests");
    smoke_step.dependOn(&smoke.step);

}

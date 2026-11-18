// SPDX-License-Identifier: Apache-2.0
// Modular contract test runner for UnifiedEngine & OpenAIApi unified routes.
// Supports `--list` and `--filter <substring>` for targeted sub-second runs.
#include "unified/harness.hpp"
#include "unified/asr_vlm_tests.hpp"
#include "unified/lifecycle_reentry_tests.hpp"
#include "unified/http_contract_tests.hpp"
#include "unified/session_history_tests.hpp"
#include "unified/mixed_output_tests.hpp"

int main(int argc, char** argv) {
  std::string filter;
  bool list_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--list") {
      list_only = true;
    } else if (arg == "--filter" && i + 1 < argc) {
      filter = argv[++i];
    }
  }

  struct Case {
    const char* name;
    void (*fn)();
  };
  const Case cases[] = {
      {"streaming_asr_to_vlm", test_streaming_asr_to_vlm},
      {"asr_partials_opt_in", test_asr_partials_opt_in},
      {"asr_partials_default_off", test_asr_partials_default_off},
      {"asr_partials_cancel", test_asr_partials_cancel},
      {"asr_partials_error", test_asr_partials_error},
      {"nonstream_asr_synthesizes_final", test_nonstream_asr_synthesizes_final},
      {"asr_plain_string_result", test_asr_plain_string_result},
      {"asr_malformed_result_fails", test_asr_malformed_result_fails},
      {"streaming_asr_malformed_final_fails", test_streaming_asr_malformed_final_fails},
      {"unknown_id_wait_fails", test_unknown_id_wait_fails},
      {"cancel_propagates_to_both_stages", test_cancel_propagates_to_both_stages},
      {"asr_failure", test_asr_failure},
      {"vlm_failure", test_vlm_failure},
      {"vlm_result_text_contract", test_vlm_result_text_contract},
      {"queue_cap_backpressure", test_queue_cap_backpressure},
      {"queue_cap_concurrent_submit", test_queue_cap_concurrent_submit},
      {"duplicate_final_and_out_of_order", test_duplicate_final_and_out_of_order},
      {"joint_image_audio", test_joint_image_audio},
      {"joint_image_remote_url", test_joint_image_remote_url},
      {"media_contract_errors", test_media_contract_errors},
      {"vlm_only_string_content", test_vlm_only_string_content},
      {"vlm_only_image_text", test_vlm_only_image_text},
      {"vlm_only_session_history", test_vlm_only_session_history},
      {"session_reuse", test_session_reuse},
      {"session_reuse_concurrent", test_session_reuse_concurrent},
      {"session_history_order", test_session_history_order},
      {"session_history_default_off", test_session_history_default_off},
      {"session_history_isolation", test_session_history_isolation},
      {"session_history_cancel_error_excluded", test_session_history_cancel_error_excluded},
      {"session_history_turn_eviction", test_session_history_turn_eviction},
      {"session_history_char_eviction", test_session_history_char_eviction},
      {"session_history_session_eviction", test_session_history_session_eviction},
      {"session_history_concurrent", test_session_history_concurrent},
      {"session_history_restart_drops", test_session_history_restart_drops},
      {"cancel_before_pumped_completion", test_cancel_before_pumped_completion},
      {"cancel_after_asr_completed_swallows_vlm", test_cancel_after_asr_completed_swallows_vlm},
      {"cancel_handoff_race_stress", test_cancel_handoff_race_stress},
      {"previous_sink_reentry_does_not_deadlock", test_previous_sink_reentry_does_not_deadlock},
      {"cancel_cannot_overtake_partial_dispatch", test_cancel_cannot_overtake_partial_dispatch},
      {"forwarded_sink_cancel_order_direct", [] { test_forwarded_sink_cancel_order(false); }},
      {"forwarded_sink_cancel_order_reentrant", [] { test_forwarded_sink_cancel_order(true); }},
      {"construction_after_pump_start_rejected", test_construction_after_pump_start_rejected},
      {"destructor_synchronizes_inflight_callback", test_destructor_synchronizes_inflight_callback},
      {"retire_wait_fallback_order", test_retire_wait_fallback_order},
      {"retired_cross_thread_order", test_retired_cross_thread_order},
      {"retired_same_thread_reentry", test_retired_same_thread_reentry},
      {"active_reentry_exception_drains", test_active_reentry_exception_drains},
      {"retired_reentry_exception_drains", test_retired_reentry_exception_drains},
      {"destructor_suppresses_previous_sink_exception", test_destructor_suppresses_previous_sink_exception},
      {"public_http_unified_sse", test_public_http_unified_sse},
      {"public_http_overload_503", test_public_http_overload_503},
      {"public_http_cancel_during_vlm", test_public_http_cancel_during_vlm},
      {"public_http_asr_model_failure", test_public_http_asr_model_failure},
      {"public_http_joint_input", test_public_http_joint_input},
      {"public_http_vlm_only", test_public_http_vlm_only},
      {"public_http_vlm_only_stream", test_public_http_vlm_only_stream},
      {"public_http_unified_errors", test_public_http_unified_errors},
      {"public_http_session_history", test_public_http_session_history},
      {"public_http_session_id_requires_unified", test_public_http_session_id_requires_unified},
      {"mixed_fanout_join_and_prompt_rules", test_mixed_fanout_join_and_prompt_rules},
      {"mixed_image_prompt_falls_back_to_answer", test_mixed_image_prompt_falls_back_to_answer},
      {"mixed_partial_leg_failure_keeps_artifacts", test_mixed_partial_leg_failure_keeps_artifacts},
      {"mixed_cancel_aborts_every_submitted_leg", test_mixed_cancel_aborts_every_submitted_leg},
      {"mixed_cancel_before_fanout_blocks_legs", test_mixed_cancel_before_fanout_blocks_legs},
      {"mixed_join_release_then_late_complete", test_mixed_join_release_then_late_complete},
      {"mixed_artifact_limits_become_leg_errors", test_mixed_artifact_limits_become_leg_errors},
      {"mixed_chunk_seq_reservation", test_mixed_chunk_seq_reservation},
      {"mixed_previous_stream_tolerates_bin_chunk", test_mixed_previous_stream_tolerates_bin_chunk},
      {"public_http_mixed_artifact_stream", test_public_http_mixed_artifact_stream},
      {"public_http_mixed_partial_failure", test_public_http_mixed_partial_failure},
      {"streamed_tts_omits_duplicate_wav_and_caps_fanout", test_streamed_tts_omits_duplicate_wav_and_caps_fanout},
      {"public_http_default_512_image_and_session_abort", test_public_http_default_512_image_and_session_abort},
      {"unified_trace_records_start_legs_and_end", test_unified_trace_records_start_legs_and_end},
  };

  if (list_only) {
    for (const Case& c : cases) std::cout << c.name << "\n";
    return 0;
  }

  alarm(120);
  for (const Case& c : cases) {
    if (!filter.empty() && std::string(c.name).find(filter) == std::string::npos) {
      continue;
    }
    c.fn();
  }
  return g_failures == 0 ? 0 : 1;
}

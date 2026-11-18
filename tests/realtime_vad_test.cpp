// SPDX-License-Identifier: Apache-2.0
// Unit checks for the realtime server's energy endpoint detector. No model, no
// socket: the VAD is pure host code, so the cases below pin its contract.
#include "serve/energy_vad.hpp"
#include "umminfer/audio/audio.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using umminfer::serve_detail::EnergyVad;
using umminfer::serve_detail::VadConfig;

int failures = 0;

void check(bool condition, const std::string& message) {
  if (condition) return;
  std::fprintf(stderr, "FAIL %s\n", message.c_str());
  ++failures;
}

// PCM16 little-endian mono 16 kHz at the given RMS level in dBFS.
std::string tone(int milliseconds, double dbfs, double hz = 440.0) {
  const int samples = milliseconds * 16;
  const double amplitude = 32767.0 * std::pow(10.0, dbfs / 20.0) * std::sqrt(2.0);
  std::string out;
  out.resize(static_cast<size_t>(samples) * 2);
  for (int i = 0; i < samples; ++i) {
    const double value = amplitude * std::sin(2.0 * 3.14159265358979323846 * hz * i / 16000.0);
    const auto sample = static_cast<int16_t>(
        value > 32767.0 ? 32767.0 : value < -32768.0 ? -32768.0 : value);
    out[static_cast<size_t>(i) * 2] = static_cast<char>(sample & 0xff);
    out[static_cast<size_t>(i) * 2 + 1] =
        static_cast<char>((static_cast<uint16_t>(sample) >> 8) & 0xff);
  }
  return out;
}

std::string silence(int milliseconds) {
  return std::string(static_cast<size_t>(milliseconds) * 32, '\0');
}

EnergyVad::Event only_event(const std::vector<EnergyVad::FrameEvent>& events,
                            size_t index) {
  return index < events.size() ? events[index].event : EnergyVad::Event::kNone;
}

void silence_never_triggers() {
  VadConfig config;
  EnergyVad vad(config);
  const std::string pcm = silence(2000);
  for (size_t offset = 0; offset < pcm.size(); offset += 3200) {
    const auto events =
        vad.push(pcm.data() + offset, std::min<size_t>(3200, pcm.size() - offset));
    check(events.empty(), "digital silence produced a VAD event");
  }
}

void speech_starts_and_reports_offset() {
  EnergyVad vad(VadConfig{});
  const std::string lead = silence(200);
  check(vad.push(lead.data(), lead.size()).empty(),
        "lead-in silence produced an event");
  const std::string voiced = tone(200, -18.0);
  const auto events = vad.push(voiced.data(), voiced.size());
  check(events.size() == 1, "speech produced more or fewer than one event");
  check(only_event(events, 0) == EnergyVad::Event::kSpeechStarted,
        "speech did not report kSpeechStarted");
  // A 10 ms frame is 320 bytes; the offset includes the frame that tripped it.
  check(!events.empty() && events[0].bytes_consumed == 320,
        "speech start offset is not the first analysis frame");
}

void silence_threshold_holds_a_turn_open() {
  VadConfig config;
  config.silence_ms = 500;
  EnergyVad vad(config);
  const std::string voiced = tone(300, -18.0);
  const auto start = vad.push(voiced.data(), voiced.size());
  check(only_event(start, 0) == EnergyVad::Event::kSpeechStarted,
        "speech was not detected before the pause");
  const std::string short_pause = silence(400);
  check(vad.push(short_pause.data(), short_pause.size()).empty(),
        "a pause shorter than silence_ms ended the turn");
  const std::string long_pause = silence(200);
  const auto stop = vad.push(long_pause.data(), long_pause.size());
  check(only_event(stop, 0) == EnergyVad::Event::kSpeechStopped,
        "silence_ms of silence did not end the turn");
}

void noise_floor_adapts_down() {
  EnergyVad vad(VadConfig{});
  // -70 dBFS is below the initial -54 dBFS gate, so it is learned as noise.
  const std::string hum = tone(1000, -70.0, 60.0);
  for (size_t offset = 0; offset < hum.size(); offset += 3200) {
    check(vad.push(hum.data() + offset,
                   std::min<size_t>(3200, hum.size() - offset))
              .empty(),
          "low-level hum produced a VAD event");
  }
  const std::string voiced = tone(200, -20.0);
  const auto events = vad.push(voiced.data(), voiced.size());
  check(only_event(events, 0) == EnergyVad::Event::kSpeechStarted,
        "speech above the adapted floor was not detected");
}

void frames_survive_split_appends() {
  EnergyVad vad(VadConfig{});
  const std::string voiced = tone(200, -18.0);
  // Odd byte counts force a frame to span two push() calls.
  const auto first = vad.push(voiced.data(), 160);
  const auto second = vad.push(voiced.data() + 160, 160);
  check(first.empty() && second.size() == 1 &&
            second[0].event == EnergyVad::Event::kSpeechStarted &&
            second[0].bytes_consumed == 160,
        "a frame split across appends was not completed exactly once");
}

void wav_pcm16_parser_extracts_headerless_samples() {
  const std::string pcm("\x01\x02\x7f\x00\x00\x80\x34\x12", 8);
  const std::string wav24k = umminfer::audio::wav_from_pcm16(pcm, 24000);
  const auto parsed = umminfer::audio::parse_wav_pcm16(wav24k);
  check(parsed.sample_rate == 24000, "24 kHz WAV sample_rate was not parsed");
  check(parsed.channels == 1, "24 kHz WAV channel count was not parsed");
  check(parsed.pcm == pcm, "parsed WAV payload kept RIFF header bytes");

  bool rejected_16k = false;
  try {
    (void)umminfer::audio::pcm16_from_wav(wav24k);
  } catch (const std::invalid_argument& err) {
    rejected_16k =
        std::string(err.what()) ==
        "WAV must contain one PCM16 mono 16 kHz data chunk";
  }
  check(rejected_16k,
        "ASR 16 kHz WAV helper did not preserve its error contract");

  bool rejected_corrupt = false;
  try {
    (void)umminfer::audio::parse_wav_pcm16(
        std::string_view("RIFF\x04\x00\x00\x00WAVE", 12));
  } catch (const std::invalid_argument&) {
    rejected_corrupt = true;
  }
  check(rejected_corrupt, "truncated WAV without fmt/data was not rejected");
}

}  // namespace

int main() {
  silence_never_triggers();
  speech_starts_and_reports_offset();
  silence_threshold_holds_a_turn_open();
  noise_floor_adapts_down();
  frames_survive_split_appends();
  wav_pcm16_parser_extracts_headerless_samples();
  if (failures != 0) {
    std::fprintf(stderr, "%d realtime VAD checks failed\n", failures);
    return 1;
  }
  std::printf("realtime VAD checks passed\n");
  return 0;
}

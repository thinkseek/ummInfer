// SPDX-License-Identifier: Apache-2.0
#include "voice.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <AudioToolbox/AudioToolbox.h>
#else
#include <dlfcn.h>
#endif

namespace umminfer::client::detail {
namespace {

[[noreturn]] void device_error(const std::string& message) {
  throw Error(ErrorCode::kConfig, message);
}

#ifndef __APPLE__

// Keep ALSA optional: ummInfer's existing build has no libasound development
// dependency. These are the stable C ABI declarations used by the voice path.
struct _snd_pcm;
struct _snd_pcm_hw_params;
using snd_pcm_t = _snd_pcm;
using snd_pcm_hw_params_t = _snd_pcm_hw_params;
using snd_pcm_uframes_t = unsigned long;
using snd_pcm_sframes_t = signed long;
using snd_pcm_stream_t = int;
constexpr snd_pcm_stream_t kPlayback = 0;
constexpr snd_pcm_stream_t kCapture = 1;
constexpr int kFormatS16Le = 2;
constexpr int kAccessRwInterleaved = 3;
constexpr int kEpipe = -32;
constexpr int kEagain = -11;

struct AlsaApi {
  void* library = nullptr;
  int (*open)(snd_pcm_t**, const char*, snd_pcm_stream_t, int) = nullptr;
  int (*set_params)(snd_pcm_t*, int, int, unsigned int, unsigned int, int, unsigned int) =
      nullptr;
  int (*prepare)(snd_pcm_t*) = nullptr;
  int (*drop)(snd_pcm_t*) = nullptr;
  int (*close)(snd_pcm_t*) = nullptr;
  int (*delay)(snd_pcm_t*, snd_pcm_sframes_t*) = nullptr;
  int (*nonblock)(snd_pcm_t*, int) = nullptr;
  int (*recover)(snd_pcm_t*, int, int) = nullptr;
  int (*hw_malloc)(snd_pcm_hw_params_t**) = nullptr;
  void (*hw_free)(snd_pcm_hw_params_t*) = nullptr;
  int (*hw_current)(snd_pcm_t*, snd_pcm_hw_params_t*) = nullptr;
  int (*hw_get_rate)(snd_pcm_hw_params_t*, unsigned int*, int*) = nullptr;
  int (*hw_get_channels)(snd_pcm_hw_params_t*, unsigned int*) = nullptr;
  int (*hw_get_format)(snd_pcm_hw_params_t*, int*) = nullptr;
  int (*hw_get_access)(snd_pcm_hw_params_t*, int*) = nullptr;
  snd_pcm_sframes_t (*readi)(snd_pcm_t*, void*, snd_pcm_uframes_t) = nullptr;
  snd_pcm_sframes_t (*writei)(snd_pcm_t*, const void*, snd_pcm_uframes_t) = nullptr;
  const char* (*strerror)(int) = nullptr;

  AlsaApi() {
    library = dlopen("libasound.so.2", RTLD_NOW | RTLD_LOCAL);
    if (!library) device_error("voice audio requires libasound.so.2 on Linux");
    load(open, "snd_pcm_open");
    load(set_params, "snd_pcm_set_params");
    load(prepare, "snd_pcm_prepare");
    load(drop, "snd_pcm_drop");
    load(close, "snd_pcm_close");
    load(delay, "snd_pcm_delay");
    load(nonblock, "snd_pcm_nonblock");
    load(recover, "snd_pcm_recover");
    load(hw_malloc, "snd_pcm_hw_params_malloc");
    load(hw_free, "snd_pcm_hw_params_free");
    load(hw_current, "snd_pcm_hw_params_current");
    load(hw_get_rate, "snd_pcm_hw_params_get_rate");
    load(hw_get_channels, "snd_pcm_hw_params_get_channels");
    load(hw_get_format, "snd_pcm_hw_params_get_format");
    load(hw_get_access, "snd_pcm_hw_params_get_access");
    load(readi, "snd_pcm_readi");
    load(writei, "snd_pcm_writei");
    load(strerror, "snd_strerror");
  }

  ~AlsaApi() {
    if (library) dlclose(library);
  }

  template <typename T>
  void load(T& target, const char* name) {
    target = reinterpret_cast<T>(dlsym(library, name));
    if (!target) device_error(std::string("voice audio backend is missing ") + name);
  }

  void verify_exact(snd_pcm_t* pcm, unsigned int expected_rate,
                    unsigned int expected_channels) const {
    snd_pcm_hw_params_t* raw = nullptr;
    int result = hw_malloc(&raw);
    if (result < 0 || !raw) device_error("cannot allocate ALSA format verification");
    std::unique_ptr<snd_pcm_hw_params_t, void (*)(snd_pcm_hw_params_t*)> params(raw, hw_free);
    unsigned int rate = 0;
    unsigned int channels = 0;
    int format = -1;
    int access = -1;
    if ((result = hw_current(pcm, raw)) < 0 ||
        (result = hw_get_rate(raw, &rate, nullptr)) < 0 ||
        (result = hw_get_channels(raw, &channels)) < 0 ||
        (result = hw_get_format(raw, &format)) < 0 ||
        (result = hw_get_access(raw, &access)) < 0) {
      device_error("cannot verify negotiated ALSA PCM format: " + message(result));
    }
    if (rate != expected_rate || channels != expected_channels || format != kFormatS16Le ||
        access != kAccessRwInterleaved) {
      device_error("ALSA negotiated a different PCM format than the realtime contract");
    }
  }

  std::string message(int error) const {
    const char* text = strerror ? strerror(error) : nullptr;
    return text ? text : "ALSA error " + std::to_string(error);
  }
};

class AlsaMicrophone final : public VoiceMicrophone {
 public:
  explicit AlsaMicrophone(const std::string& device) : device_(device.empty() ? "default" : device) {}
  ~AlsaMicrophone() override { stop(); }

  void start(PcmCallback on_pcm, ErrorCallback on_error) override {
    if (running_) return;
    on_pcm_ = std::move(on_pcm);
    on_error_ = std::move(on_error);
    const int opened = api_.open(&pcm_, device_.c_str(), kCapture, 0);
    if (opened < 0) {
      device_error("cannot open voice microphone: " + api_.message(opened));
    }
    int result = api_.set_params(pcm_, kFormatS16Le, kAccessRwInterleaved, 1, 16000, 0, 40000);
    if (result < 0) {
      device_error("voice microphone does not support 16 kHz mono PCM16: " +
                   api_.message(result));
    }
    api_.verify_exact(pcm_, 16000, 1);
    result = api_.nonblock(pcm_, 1);
    if (result < 0) device_error("cannot make microphone polling nonblocking");
    running_ = true;
    worker_ = std::thread([this] { run(); });
  }

  void stop() noexcept override {
    running_ = false;
    // readi() is nonblocking: only the capture worker touches the live handle;
    // close happens after join, never concurrently with readi/recover.
    if (worker_.joinable() && worker_.get_id() != std::this_thread::get_id()) worker_.join();
    if (pcm_) {
      api_.close(pcm_);
      pcm_ = nullptr;
    }
    on_pcm_ = nullptr;
    on_error_ = nullptr;
  }

 private:
  void run() noexcept {
    constexpr snd_pcm_uframes_t kFrames = 320;  // 20 ms at 16 kHz
    std::vector<int16_t> samples(kFrames);
    while (running_) {
      const snd_pcm_sframes_t got = api_.readi(pcm_, samples.data(), kFrames);
      if (got < 0) {
        if (!running_) break;
        if (got == kEagain) {
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
          continue;
        }
        const int recovered = api_.recover(pcm_, static_cast<int>(got), 1);
        if (recovered < 0) {
          try {
            device_error("voice microphone read failed: " + api_.message(recovered));
          } catch (...) {
            if (on_error_) on_error_(std::current_exception());
          }
          break;
        }
        continue;  // Overrun samples are discarded, never fabricated.
      }
      if (got == 0) continue;
      std::string pcm(static_cast<size_t>(got) * 2, '\0');
      std::memcpy(pcm.data(), samples.data(), pcm.size());
      if (on_pcm_) on_pcm_(std::move(pcm));
    }
  }

  AlsaApi api_;
  std::string device_;
  snd_pcm_t* pcm_ = nullptr;
  std::atomic<bool> running_{false};
  std::thread worker_;
  PcmCallback on_pcm_;
  ErrorCallback on_error_;
};

class AlsaOutput final : public VoiceAudioOutput {
 public:
  explicit AlsaOutput(const std::string& device) : device_(device.empty() ? "default" : device) {
    worker_ = std::thread([this] { run(); });
  }
  ~AlsaOutput() override { close(); }

  void enqueue(VoicePcmChunk chunk, Completion completion) override {
    if (chunk.sample_rate <= 0 || chunk.channels <= 0 || chunk.pcm.empty()) return;
    Item item;
    item.sample_rate = chunk.sample_rate;
    item.channels = chunk.channels;
    item.pcm = std::move(chunk.pcm);
    item.frames = item.pcm.size() / (static_cast<size_t>(chunk.channels) * 2);
    item.completion = std::move(completion);
    {
      std::lock_guard lock(mu_);
      if (closed_) return;
      item.generation = generation_;
      queued_.push_back(std::move(item));
    }
    cv_.notify_all();
  }

  void stop() noexcept override {
    {
      std::lock_guard lock(mu_);
      ++generation_;
      queued_.clear();
      written_.clear();
      written_frames_ = 0;
      reset_ = true;
    }
    // The nonblocking writer observes reset_ within one poll interval. No ALSA
    // call is made concurrently with the worker's writei/delay/recover calls.
    cv_.notify_all();
  }

  void close() noexcept override {
    {
      std::lock_guard lock(mu_);
      if (closed_ && !worker_.joinable()) return;
      closed_ = true;
      ++generation_;
      queued_.clear();
      written_.clear();
    }
    cv_.notify_all();
    if (worker_.joinable() && worker_.get_id() != std::this_thread::get_id()) worker_.join();
    if (pcm_) {
      api_.close(pcm_);
      pcm_ = nullptr;
    }
  }

 private:
  struct Item {
    std::string pcm;
    int sample_rate = 0;
    int channels = 0;
    size_t frames = 0;
    uint64_t generation = 0;
    uint64_t end_frame = 0;
    Completion completion;
  };

  // Called only by the output worker, before writing an item.
  void configure(int rate, int channels) {
    if (pcm_ && rate == rate_ && channels == channels_) return;
    if (pcm_) {
      bool busy = false;
      {
        std::lock_guard lock(mu_);
        busy = !written_.empty();
      }
      if (busy) device_error("voice output PCM format changed while audio was pending");
      api_.close(pcm_);
      pcm_ = nullptr;
    }
    const int opened = api_.open(&pcm_, device_.c_str(), kPlayback, 0);
    if (opened < 0) device_error("cannot open voice speaker: " + api_.message(opened));
    int result = api_.set_params(pcm_, kFormatS16Le, kAccessRwInterleaved,
                                 static_cast<unsigned int>(channels),
                                 static_cast<unsigned int>(rate), 0, 60000);
    if (result < 0) {
      device_error("voice speaker does not support the response PCM16 format: " +
                   api_.message(result));
    }
    api_.verify_exact(pcm_, static_cast<unsigned int>(rate),
                      static_cast<unsigned int>(channels));
    result = api_.nonblock(pcm_, 1);
    if (result < 0) device_error("cannot make speaker polling nonblocking");
    result = api_.prepare(pcm_);
    if (result < 0) device_error("cannot prepare voice speaker: " + api_.message(result));
    rate_ = rate;
    channels_ = channels;
    {
      std::lock_guard lock(mu_);
      reset_ = false;
      written_frames_ = 0;
    }
  }

  void update_completions() {
    if (!pcm_) return;
    std::vector<Completion> completed;
    bool consumed = true;
    {
      std::lock_guard lock(mu_);
      if (written_.empty()) return;
    }
    snd_pcm_sframes_t delay = 0;
    const int status = api_.delay(pcm_, &delay);
    {
      std::lock_guard lock(mu_);
      if (written_.empty()) return;
      if (status < 0) {
        // Device removal/xrun cannot establish actual consumption. Complete
        // every affected item as failed so the controller closes/reconnects
        // instead of retaining the response forever waiting for a final ACK.
        consumed = false;
        while (!written_.empty()) {
          if (written_.front().completion) {
            completed.push_back(std::move(written_.front().completion));
          }
          written_.pop_front();
        }
        written_frames_ = 0;
        reset_ = true;
      } else {
        const uint64_t pending_delay = delay > 0 ? static_cast<uint64_t>(delay) : 0;
        const uint64_t played = written_frames_ > pending_delay
                                    ? written_frames_ - pending_delay
                                    : 0;
        while (!written_.empty() && written_.front().end_frame <= played) {
          if (written_.front().completion) {
            completed.push_back(std::move(written_.front().completion));
          }
          written_.pop_front();
        }
      }
    }
    for (auto& completion : completed) completion(consumed);
  }

  void run() noexcept {
    while (true) {
      Item item;
      bool have_item = false;
      {
        std::unique_lock lock(mu_);
        if (closed_) return;
        if (reset_) {
          if (pcm_) {
            api_.drop(pcm_);
            api_.prepare(pcm_);
          }
          reset_ = false;
          written_frames_ = 0;
        }
        if (queued_.empty()) {
          if (written_.empty()) {
            cv_.wait(lock, [this] { return closed_ || reset_ || !queued_.empty(); });
          } else {
            cv_.wait_for(lock, std::chrono::milliseconds(2), [this] {
              return closed_ || reset_ || !queued_.empty();
            });
          }
          if (closed_) return;
          if (reset_ || queued_.empty()) {
            lock.unlock();
            update_completions();
            continue;
          }
        }
        item = std::move(queued_.front());
        queued_.pop_front();
        have_item = true;
      }

      if (!have_item) continue;
      bool interrupted = false;
      {
        std::lock_guard lock(mu_);
        interrupted = closed_ || reset_ || item.generation != generation_;
      }
      if (interrupted) continue;
      try {
        configure(item.sample_rate, item.channels);
      } catch (...) {
        if (item.completion) item.completion(false);
        return;
      }
      size_t offset = 0;
      while (offset < item.frames) {
        {
          std::lock_guard lock(mu_);
          interrupted = closed_ || reset_ || item.generation != generation_;
        }
        if (interrupted) break;
        const snd_pcm_sframes_t wrote = api_.writei(
            pcm_, item.pcm.data() + offset * static_cast<size_t>(channels_) * 2,
            static_cast<snd_pcm_uframes_t>(item.frames - offset));
        if (wrote > 0) {
          offset += static_cast<size_t>(wrote);
          continue;
        }
        {
          std::lock_guard lock(mu_);
          interrupted = closed_ || reset_ || item.generation != generation_;
        }
        if (interrupted) break;
        if (wrote == kEagain) {
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
          continue;
        }
        const int error = wrote == 0 ? kEpipe : static_cast<int>(wrote);
        const int recovered = api_.recover(pcm_, error, 1);
        if (recovered < 0) {
          if (item.completion) item.completion(false);
          return;
        }
      }
      {
        std::lock_guard lock(mu_);
        interrupted = interrupted || closed_ || reset_ || item.generation != generation_;
        if (!interrupted && offset == item.frames) {
          written_frames_ += item.frames;
          item.end_frame = written_frames_;
          written_.push_back(std::move(item));
        }
      }
      update_completions();
    }
  }

  AlsaApi api_;
  std::string device_;
  snd_pcm_t* pcm_ = nullptr;
  int rate_ = 0;
  int channels_ = 0;
  std::mutex mu_;
  std::condition_variable cv_;
  std::thread worker_;
  std::deque<Item> queued_;
  std::deque<Item> written_;
  uint64_t generation_ = 0;
  uint64_t written_frames_ = 0;
  bool reset_ = false;
  bool closed_ = false;
};

#else  // __APPLE__

AudioStreamBasicDescription pcm_description(int rate, int channels) {
  AudioStreamBasicDescription format = {};
  format.mSampleRate = rate;
  format.mFormatID = kAudioFormatLinearPCM;
  format.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
  format.mBitsPerChannel = 16;
  format.mBytesPerPacket = static_cast<UInt32>(channels * 2);
  format.mFramesPerPacket = 1;
  format.mBytesPerFrame = static_cast<UInt32>(channels * 2);
  format.mChannelsPerFrame = static_cast<UInt32>(channels);
  return format;
}

void check_audio(OSStatus status, const char* operation) {
  if (status != noErr) {
    device_error(std::string(operation) + " failed with CoreAudio status " +
                 std::to_string(static_cast<int>(status)));
  }
}

class QueueMicrophone final : public VoiceMicrophone {
 public:
  explicit QueueMicrophone(const std::string& device) {
    if (!device.empty()) device_error("named macOS voice input devices are not supported");
  }
  ~QueueMicrophone() override { stop(); }

  void start(PcmCallback on_pcm, ErrorCallback on_error) override {
    if (queue_) return;
    on_pcm_ = std::move(on_pcm);
    on_error_ = std::move(on_error);
    AudioStreamBasicDescription format = pcm_description(16000, 1);
    check_audio(AudioQueueNewInput(&format, &QueueMicrophone::callback, this, nullptr, nullptr, 0,
                                   &queue_),
                "open 16 kHz mono PCM16 microphone");
    try {
      for (AudioQueueBufferRef& buffer : buffers_) {
        check_audio(AudioQueueAllocateBuffer(queue_, 640, &buffer), "allocate microphone buffer");
        check_audio(AudioQueueEnqueueBuffer(queue_, buffer, 0, nullptr), "queue microphone buffer");
      }
      check_audio(AudioQueueStart(queue_, nullptr), "start microphone");
    } catch (...) {
      stop();
      throw;
    }
  }

  void stop() noexcept override {
    AudioQueueRef queue = queue_;
    queue_ = nullptr;
    if (queue) {
      AudioQueueStop(queue, true);
      AudioQueueDispose(queue, true);
    }
    buffers_.fill(nullptr);
    on_pcm_ = nullptr;
    on_error_ = nullptr;
  }

 private:
  static void callback(void* opaque, AudioQueueRef queue, AudioQueueBufferRef buffer,
                       const AudioTimeStamp*, UInt32, const AudioStreamPacketDescription*) {
    auto* self = static_cast<QueueMicrophone*>(opaque);
    if (!self->queue_ || self->queue_ != queue || !self->on_pcm_) return;
    if (buffer->mAudioDataByteSize > 0) {
      self->on_pcm_(std::string(static_cast<const char*>(buffer->mAudioData),
                                buffer->mAudioDataByteSize));
    }
    if (self->queue_ == queue && AudioQueueEnqueueBuffer(queue, buffer, 0, nullptr) != noErr &&
        self->on_error_) {
      self->on_error_(std::make_exception_ptr(
          Error(ErrorCode::kConfig, "cannot requeue the macOS microphone buffer")));
    }
  }

  AudioQueueRef queue_ = nullptr;
  std::array<AudioQueueBufferRef, 3> buffers_{};
  PcmCallback on_pcm_;
  ErrorCallback on_error_;
};

class QueueOutput final : public VoiceAudioOutput {
 public:
  explicit QueueOutput(const std::string& device) {
    if (!device.empty()) device_error("named macOS voice output devices are not supported");
  }
  ~QueueOutput() override { close(); }

  void enqueue(VoicePcmChunk chunk, Completion completion) override {
    std::lock_guard lock(mu_);
    if (closed_) return;
    if (queue_ && (rate_ != chunk.sample_rate || channels_ != chunk.channels)) {
      device_error("voice output PCM format changed without a playback reset");
    }
    if (!queue_) {
      rate_ = chunk.sample_rate;
      channels_ = chunk.channels;
      AudioStreamBasicDescription format = pcm_description(rate_, channels_);
      check_audio(AudioQueueNewOutput(&format, &QueueOutput::callback, this, nullptr, nullptr, 0,
                                      &queue_),
                  "open voice speaker");
      check_audio(AudioQueueStart(queue_, nullptr), "start voice speaker");
    }
    AudioQueueBufferRef buffer = nullptr;
    check_audio(AudioQueueAllocateBuffer(queue_, static_cast<UInt32>(chunk.pcm.size()), &buffer),
                "allocate voice playback buffer");
    std::memcpy(buffer->mAudioData, chunk.pcm.data(), chunk.pcm.size());
    buffer->mAudioDataByteSize = static_cast<UInt32>(chunk.pcm.size());
    auto context = std::make_unique<BufferContext>();
    context->generation = generation_;
    context->completion = std::move(completion);
    contexts_[buffer] = std::move(context);
    const OSStatus status = AudioQueueEnqueueBuffer(queue_, buffer, 0, nullptr);
    if (status != noErr) {
      contexts_.erase(buffer);
      AudioQueueFreeBuffer(queue_, buffer);
      check_audio(status, "queue voice playback");
    }
  }

  void stop() noexcept override {
    AudioQueueRef queue = nullptr;
    {
      std::lock_guard lock(mu_);
      ++generation_;
      queue = queue_;
      queue_ = nullptr;
      contexts_.clear();
    }
    if (queue) {
      AudioQueueStop(queue, true);
      AudioQueueDispose(queue, true);  // Also releases its allocated buffers.
    }
  }

  void close() noexcept override {
    stop();
    std::lock_guard lock(mu_);
    closed_ = true;
  }

 private:
  struct BufferContext {
    uint64_t generation = 0;
    Completion completion;
  };

  static void callback(void* opaque, AudioQueueRef queue, AudioQueueBufferRef buffer) {
    auto* self = static_cast<QueueOutput*>(opaque);
    Completion completion;
    {
      std::lock_guard lock(self->mu_);
      auto found = self->contexts_.find(buffer);
      if (found == self->contexts_.end()) return;
      if (self->queue_ == queue && found->second->generation == self->generation_) {
        completion = std::move(found->second->completion);
      }
      self->contexts_.erase(found);
      AudioQueueFreeBuffer(queue, buffer);
    }
    // CoreAudio's buffer callback is the actual consumption boundary.
    if (completion) completion(true);
  }

  std::mutex mu_;
  AudioQueueRef queue_ = nullptr;
  int rate_ = 0;
  int channels_ = 0;
  uint64_t generation_ = 0;
  bool closed_ = false;
  std::map<AudioQueueBufferRef, std::unique_ptr<BufferContext>> contexts_;
};

#endif

}  // namespace

std::unique_ptr<VoiceMicrophone> make_voice_microphone(const std::string& device) {
#ifndef __APPLE__
  return std::make_unique<AlsaMicrophone>(device);
#else
  return std::make_unique<QueueMicrophone>(device);
#endif
}

std::unique_ptr<VoiceAudioOutput> make_voice_audio_output(const std::string& device) {
#ifndef __APPLE__
  return std::make_unique<AlsaOutput>(device);
#else
  return std::make_unique<QueueOutput>(device);
#endif
}

}  // namespace umminfer::client::detail

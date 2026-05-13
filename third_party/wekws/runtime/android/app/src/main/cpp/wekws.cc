#include <jni.h>

#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "frontend/feature_pipeline.h"
#include "kws/keyword_spotting.h"
#include "utils/log.h"

namespace wekws {
constexpr int kSampleRate = 16000;
constexpr int kFbankDim = 80;
constexpr int kContextLeft = 2;
constexpr int kContextRight = 2;
constexpr int kContextWindow = kContextLeft + kContextRight + 1;
constexpr int kModelInputDim = kFbankDim * kContextWindow;
constexpr int kFrameSkip = 3;
constexpr int kReadFrames = 80;
constexpr int kBlankId = 0;
constexpr int kScoreBeamSize = 3;
constexpr int kPathBeamSize = 20;
constexpr int kMinFrames = 5;
constexpr int kMaxFrames = 250;
constexpr int kIntervalFrames = 50;
constexpr float kScorePruneThreshold = 0.05f;
constexpr float kSpeechRmsThreshold = 1200.0f;
constexpr int kSpeechPeakThreshold = 5000;
constexpr int kSilenceChunksBeforeReset = 20;
constexpr bool kDebugDecode = true;
const int kKeywordIds[] = {1, 2, 3, 4};

struct Node {
  int token = 0;
  int frame = 0;
  float prob = 0.0f;
};

struct Hyp {
  std::vector<int> prefix;
  double pb = 0.0;
  double pnb = 0.0;
  std::vector<Node> nodes;
};

struct BeamState {
  double pb = 0.0;
  double pnb = 0.0;
  std::vector<Node> nodes;
};

std::shared_ptr<KeywordSpotting> spotter;
std::shared_ptr<wenet::FeaturePipelineConfig> feature_config;
std::shared_ptr<wenet::FeaturePipeline> feature_pipeline;

std::mutex pipeline_mutex;
std::mutex result_mutex;
std::mutex debug_mutex;
std::string result;
std::string debug_result;
float threshold = 0.40f;
int total_frames = 0;
int last_active_frame = -1;
int feats_ctx_offset = 0;
std::vector<std::vector<float>> feature_remained;
std::vector<Hyp> cur_hyps;
int debug_chunk_count = 0;
int last_pruned_hyps = 0;
int consecutive_silence_chunks = 0;

std::string TokenToString(int token) {
  switch (token) {
    case 0:
      return "<blk>";
    case 1:
      return "an";
    case 2:
      return "ju";
    case 3:
      return "xiao";
    case 4:
      return "bao";
    case 5:
      return "<filler>";
    default:
      return std::to_string(token);
  }
}

std::string PrefixToString(const std::vector<int>& prefix) {
  std::ostringstream oss;
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (i > 0) oss << ",";
    oss << TokenToString(prefix[i]);
  }
  return oss.str();
}

std::vector<float> ConcatContext(
    const std::vector<std::vector<float>>& feats, int start) {
  std::vector<float> out;
  out.reserve(kModelInputDim);
  for (int i = 0; i < kContextWindow; ++i) {
    const auto& frame = feats[start + i];
    out.insert(out.end(), frame.begin(), frame.end());
  }
  return out;
}

std::vector<std::vector<float>> ExpandAndSkip(
    const std::vector<std::vector<float>>& feats) {
  if (feats.empty()) return {};

  std::vector<std::vector<float>> padded;
  if (feature_remained.empty()) {
    for (int i = 0; i < kContextLeft; ++i) {
      padded.push_back(feats.front());
    }
    padded.insert(padded.end(), feats.begin(), feats.end());
  } else {
    padded = feature_remained;
    padded.insert(padded.end(), feats.begin(), feats.end());
  }

  int ctx_frames = static_cast<int>(padded.size()) -
                   (kContextRight + kContextRight);
  std::vector<std::vector<float>> expanded;
  for (int i = 0; i < ctx_frames; ++i) {
    expanded.push_back(ConcatContext(padded, i));
  }

  feature_remained.clear();
  int remain = kContextLeft + kContextRight;
  int start = std::max(0, static_cast<int>(feats.size()) - remain);
  feature_remained.insert(feature_remained.end(), feats.begin() + start,
                          feats.end());

  if (expanded.empty()) return expanded;
  int last_remainder =
      feats_ctx_offset == 0 ? 0 : kFrameSkip - feats_ctx_offset;
  int remainder =
      (static_cast<int>(expanded.size()) + last_remainder) % kFrameSkip;
  std::vector<std::vector<float>> skipped;
  for (int i = feats_ctx_offset; i < static_cast<int>(expanded.size());
       i += kFrameSkip) {
    skipped.push_back(std::move(expanded[i]));
  }
  feats_ctx_offset = remainder == 0 ? 0 : kFrameSkip - remainder;
  return skipped;
}

void ResetDecoder() {
  cur_hyps.clear();
  cur_hyps.push_back(Hyp{{}, 1.0, 0.0, {}});
}

void ResetStreamingState(bool reset_model_cache, bool reset_feature_pipeline) {
  if (reset_model_cache && spotter) spotter->Reset();
  if (reset_feature_pipeline && feature_pipeline) feature_pipeline->Reset();
  total_frames = 0;
  last_active_frame = -1;
  feats_ctx_offset = 0;
  feature_remained.clear();
  debug_result.clear();
  debug_chunk_count = 0;
  last_pruned_hyps = 0;
  ResetDecoder();
}

bool IsSpeechChunk(const std::vector<int16_t>& data) {
  if (data.empty()) return false;
  double sum_squares = 0.0;
  int peak = 0;
  for (int16_t sample : data) {
    int value = static_cast<int>(sample);
    int abs_value = std::abs(value);
    peak = std::max(peak, abs_value);
    sum_squares += static_cast<double>(value) * value;
  }
  double rms = std::sqrt(sum_squares / static_cast<double>(data.size()));
  return rms >= kSpeechRmsThreshold || peak >= kSpeechPeakThreshold;
}

bool IsKeywordToken(int token) {
  return token == kBlankId || token == kKeywordIds[0] ||
         token == kKeywordIds[1] || token == kKeywordIds[2] ||
         token == kKeywordIds[3];
}

bool NearlyZero(double value) { return std::abs(value) <= 0.000001; }

std::vector<Hyp> CtcPrefixBeamSearch(int frame,
                                     const std::vector<float>& probs) {
  if (probs.empty()) return cur_hyps;

  std::vector<std::pair<float, int>> ranked;
  ranked.reserve(probs.size());
  for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
    ranked.emplace_back(probs[i], i);
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<int> filter_index;
  int top_k = std::min(kScoreBeamSize, static_cast<int>(ranked.size()));
  for (int i = 0; i < top_k; ++i) {
    float prob = ranked[i].first;
    int token = ranked[i].second;
    if (prob > kScorePruneThreshold && IsKeywordToken(token)) {
      filter_index.push_back(token);
    }
  }
  if (filter_index.empty()) return cur_hyps;

  std::map<std::vector<int>, BeamState> next_hyps;

  for (int s : filter_index) {
    double ps = probs[s];
    for (const auto& hyp : cur_hyps) {
      const std::vector<int>& prefix = hyp.prefix;
      double pb = hyp.pb;
      double pnb = hyp.pnb;
      int last = prefix.empty() ? -1 : prefix.back();

      if (s == kBlankId) {
        BeamState& state = next_hyps[prefix];
        state.pb += pb * ps + pnb * ps;
        state.nodes = hyp.nodes;
      } else if (s == last) {
        if (!NearlyZero(pnb)) {
          BeamState& state = next_hyps[prefix];
          state.pnb += pnb * ps;
          state.nodes = hyp.nodes;
          if (!state.nodes.empty() && ps > state.nodes.back().prob) {
            state.nodes.back().prob = static_cast<float>(ps);
            state.nodes.back().frame = frame;
          }
        }

        if (!NearlyZero(pb)) {
          std::vector<int> n_prefix = prefix;
          n_prefix.push_back(s);
          BeamState& state = next_hyps[n_prefix];
          state.pnb += pb * ps;
          state.nodes = hyp.nodes;
          state.nodes.push_back(Node{s, frame, static_cast<float>(ps)});
        }
      } else {
        std::vector<int> n_prefix = prefix;
        n_prefix.push_back(s);
        BeamState& state = next_hyps[n_prefix];
        state.pnb += pb * ps + pnb * ps;
        if (!state.nodes.empty()) {
          if (ps > state.nodes.back().prob) {
            state.nodes.pop_back();
            state.nodes.push_back(Node{s, frame, static_cast<float>(ps)});
          }
        } else {
          state.nodes = hyp.nodes;
          state.nodes.push_back(Node{s, frame, static_cast<float>(ps)});
        }
      }
    }
  }

  std::vector<Hyp> out;
  out.reserve(next_hyps.size());
  for (const auto& item : next_hyps) {
    out.push_back(Hyp{item.first, item.second.pb, item.second.pnb,
                      item.second.nodes});
  }
  std::sort(out.begin(), out.end(), [](const Hyp& a, const Hyp& b) {
    return (a.pb + a.pnb) > (b.pb + b.pnb);
  });
  if (out.size() > kPathBeamSize) out.resize(kPathBeamSize);
  return out;
}

void PruneStalePartialPrefixes(int frame) {
  last_pruned_hyps = 0;
  if (cur_hyps.empty()) {
    ResetDecoder();
    return;
  }

  std::vector<Hyp> kept;
  kept.reserve(cur_hyps.size());
  for (const auto& hyp : cur_hyps) {
    if (hyp.prefix.empty()) {
      kept.push_back(hyp);
      continue;
    }
    if (hyp.nodes.empty()) {
      kept.push_back(hyp);
      continue;
    }

    int start = hyp.nodes.front().frame;
    bool stale_by_total_span = frame - start > kMaxFrames;
    if (!stale_by_total_span) {
      kept.push_back(hyp);
    } else {
      ++last_pruned_hyps;
    }
  }

  if (kept.empty()) {
    ResetDecoder();
    return;
  }
  std::sort(kept.begin(), kept.end(), [](const Hyp& a, const Hyp& b) {
    return (a.pb + a.pnb) > (b.pb + b.pnb);
  });
  if (kept.size() > kPathBeamSize) kept.resize(kPathBeamSize);
  cur_hyps = std::move(kept);
}

int FindKeywordOffset(const std::vector<int>& prefix) {
  constexpr int keyword_len = 4;
  if (prefix.size() < keyword_len) return -1;
  for (int i = 0; i <= static_cast<int>(prefix.size()) - keyword_len; ++i) {
    bool matched = true;
    for (int j = 0; j < keyword_len; ++j) {
      if (prefix[i + j] != kKeywordIds[j]) {
        matched = false;
        break;
      }
    }
    if (matched) return i;
  }
  return -1;
}

struct DetectionResult {
  bool has_keyword = false;
  bool wakeup = false;
  int start = 0;
  int end = 0;
  float score = 0.0f;
};

DetectionResult ExecuteDetection() {
  DetectionResult result;
  for (const auto& hyp : cur_hyps) {
    int offset = FindKeywordOffset(hyp.prefix);
    if (offset < 0) continue;
    if (offset + 3 >= static_cast<int>(hyp.nodes.size())) continue;

    float product = 1.0f;
    for (int i = offset; i < offset + 4; ++i) {
      product *= std::max(1e-6f, hyp.nodes[i].prob);
    }
    float score = std::sqrt(product);
    if (!result.has_keyword || score > result.score) {
      result.has_keyword = true;
      result.start = hyp.nodes[offset].frame;
      result.end = hyp.nodes[offset + 3].frame;
      result.score = score;
    }
  }

  if (result.has_keyword) {
    int duration = result.end - result.start;
    result.wakeup = result.score >= threshold &&
                    kMinFrames <= duration && duration <= kMaxFrames &&
                    (last_active_frame == -1 ||
                     result.end - last_active_frame >= kIntervalFrames);
  }
  return result;
}

std::string FormatResult(bool wakeup, float score, double elapsed_ms) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  if (wakeup) {
    oss << "WAKEUP keyword=anju_xiaobao";
  } else {
    oss << "listening";
  }
  oss << " score=" << score << " threshold=" << threshold
      << " frame=" << total_frames << " infer_ms=" << elapsed_ms;
  return oss.str();
}

void init(JNIEnv* env, jobject, jstring jModelDir, jfloat jThreshold) {
  const char* pModelDir = env->GetStringUTFChars(jModelDir, nullptr);
  std::string model_dir(pModelDir);
  env->ReleaseStringUTFChars(jModelDir, pModelDir);

  threshold = jThreshold;
  KeywordSpotting::InitEngineThreads(2);
  spotter = std::make_shared<KeywordSpotting>(model_dir + "/kws.onnx");
  feature_config =
      std::make_shared<wenet::FeaturePipelineConfig>(kFbankDim, kSampleRate);
  feature_pipeline = std::make_shared<wenet::FeaturePipeline>(*feature_config);
  feature_config->Info();

  consecutive_silence_chunks = 0;
  ResetStreamingState(false, false);
  {
    std::lock_guard<std::mutex> lock(result_mutex);
    result = FormatResult(false, 0.0f, 0.0);
  }
}

void reset(JNIEnv*, jobject) {
  std::lock_guard<std::mutex> lock(pipeline_mutex);
  consecutive_silence_chunks = 0;
  ResetStreamingState(true, true);
  {
    std::lock_guard<std::mutex> lock(result_mutex);
    result = FormatResult(false, 0.0f, 0.0);
  }
}

void set_threshold(JNIEnv*, jobject, jfloat jThreshold) { threshold = jThreshold; }

void accept_waveform(JNIEnv* env, jobject, jshortArray jWaveform) {
  if (!feature_pipeline) return;
  jsize size = env->GetArrayLength(jWaveform);
  int16_t* waveform = env->GetShortArrayElements(jWaveform, nullptr);
  std::vector<int16_t> data(waveform, waveform + size);
  env->ReleaseShortArrayElements(jWaveform, waveform, JNI_ABORT);

  std::lock_guard<std::mutex> lock(pipeline_mutex);
  bool speech = IsSpeechChunk(data);
  if (speech) {
    if (consecutive_silence_chunks >= kSilenceChunksBeforeReset) {
      ResetStreamingState(true, true);
      {
        std::lock_guard<std::mutex> debug_lock(debug_mutex);
        std::ostringstream dbg;
        dbg << "speech_onset_reset silence_chunks="
            << consecutive_silence_chunks;
        debug_result = dbg.str();
      }
      __android_log_print(ANDROID_LOG_INFO, "WEKWS_NATIVE",
                          "speech_onset_reset silence_chunks=%d",
                          consecutive_silence_chunks);
    }
    consecutive_silence_chunks = 0;
  } else {
    ++consecutive_silence_chunks;
  }
  feature_pipeline->AcceptWaveform(data);
}

void set_input_finished() {
  std::lock_guard<std::mutex> lock(pipeline_mutex);
  if (feature_pipeline && !feature_pipeline->input_finished()) {
    feature_pipeline->set_input_finished();
  }
}

void start_spot() {
  if (!spotter || !feature_pipeline) return;

  std::vector<std::vector<float>> raw_feats;
  if (!feature_pipeline->Read(kReadFrames, &raw_feats)) return;
  auto start = std::chrono::steady_clock::now();
  std::vector<std::vector<float>> feats = ExpandAndSkip(raw_feats);
  std::vector<std::vector<float>> probs;
  spotter->Forward(feats, &probs);

  float best_keyword_score = 0.0f;
  bool wakeup = false;
  std::ostringstream top_trace;
  int trace_frames = 0;
  for (int idx = 0; idx < static_cast<int>(probs.size()); ++idx) {
    int frame = total_frames + idx * kFrameSkip;
    if (kDebugDecode && trace_frames < 8 && !probs[idx].empty()) {
      std::vector<std::pair<float, int>> ranked;
      ranked.reserve(probs[idx].size());
      for (int i = 0; i < static_cast<int>(probs[idx].size()); ++i) {
        ranked.emplace_back(probs[idx][i], i);
      }
      std::sort(ranked.begin(), ranked.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      top_trace << " f" << frame << ":";
      for (int k = 0; k < std::min(3, static_cast<int>(ranked.size())); ++k) {
        top_trace << TokenToString(ranked[k].second) << "=" << std::fixed
                  << std::setprecision(2) << ranked[k].first << "/";
      }
      ++trace_frames;
    }
    cur_hyps = CtcPrefixBeamSearch(frame, probs[idx]);
    DetectionResult detection = ExecuteDetection();
    best_keyword_score = std::max(best_keyword_score, detection.score);
    if (detection.wakeup) {
      wakeup = true;
      last_active_frame = detection.end;
      ResetDecoder();
      break;
    }
    PruneStalePartialPrefixes(frame);
  }
  total_frames += static_cast<int>(probs.size()) * kFrameSkip;

  if (kDebugDecode && (++debug_chunk_count % 5 == 0 || wakeup ||
                       best_keyword_score > 0.0f)) {
    std::string top_path = cur_hyps.empty() ? "" : PrefixToString(cur_hyps[0].prefix);
    double top_path_score = cur_hyps.empty() ? 0.0 : cur_hyps[0].pb + cur_hyps[0].pnb;
    std::ostringstream dbg;
    dbg << "decode_debug raw_feats=" << raw_feats.size()
        << " expanded_feats=" << feats.size()
        << " out_frames=" << probs.size()
        << " best_keyword_score=" << best_keyword_score
        << " top_path=" << top_path
        << " top_path_score=" << top_path_score
        << " pruned_hyps=" << last_pruned_hyps
        << " trace=" << top_trace.str();
    __android_log_print(ANDROID_LOG_INFO, "WEKWS_NATIVE", "%s",
                        dbg.str().c_str());
    {
      std::lock_guard<std::mutex> lock(debug_mutex);
      debug_result = dbg.str();
    }
  }

  auto end = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  {
    std::lock_guard<std::mutex> lock(result_mutex);
    result = FormatResult(wakeup, best_keyword_score, elapsed_ms);
  }
}

jstring get_result(JNIEnv* env, jobject) {
  std::lock_guard<std::mutex> lock(result_mutex);
  return env->NewStringUTF(result.c_str());
}

jstring get_debug(JNIEnv* env, jobject) {
  std::lock_guard<std::mutex> lock(debug_mutex);
  return env->NewStringUTF(debug_result.c_str());
}
}  // namespace wekws

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }
  jclass c = env->FindClass("cn/org/wenet/wekws/Spot");
  if (c == nullptr) return JNI_ERR;

  static const JNINativeMethod methods[] = {
      {"init", "(Ljava/lang/String;F)V", reinterpret_cast<void*>(wekws::init)},
      {"reset", "()V", reinterpret_cast<void*>(wekws::reset)},
      {"setThreshold", "(F)V", reinterpret_cast<void*>(wekws::set_threshold)},
      {"acceptWaveform", "([S)V",
       reinterpret_cast<void*>(wekws::accept_waveform)},
      {"setInputFinished", "()V",
       reinterpret_cast<void*>(wekws::set_input_finished)},
      {"startSpot", "()V", reinterpret_cast<void*>(wekws::start_spot)},
      {"getResult", "()Ljava/lang/String;",
       reinterpret_cast<void*>(wekws::get_result)},
      {"getDebug", "()Ljava/lang/String;",
       reinterpret_cast<void*>(wekws::get_debug)},
  };
  int rc = env->RegisterNatives(c, methods,
                                sizeof(methods) / sizeof(JNINativeMethod));
  if (rc != JNI_OK) return rc;
  return JNI_VERSION_1_6;
}

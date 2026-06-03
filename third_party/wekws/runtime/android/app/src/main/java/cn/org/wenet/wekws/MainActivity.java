package cn.org.wenet.wekws;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.media.AudioManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.ToneGenerator;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MainActivity extends AppCompatActivity {

    private static final int MY_PERMISSIONS_RECORD_AUDIO = 1;
    private static final String LOG_TAG = "WEKWS";
    private static final int SAMPLE_RATE = 16000;
    private static final int READ_CHUNK_SAMPLES = 640;  // 40 ms at 16 kHz.
    private static final int MAX_QUEUE_SIZE = 100;
    private static final float DEFAULT_THRESHOLD = 0.40f;
    private static final float DEFAULT_SPEECH_RMS_THRESHOLD = 70.0f;
    private static final int DEFAULT_SPEECH_PEAK_THRESHOLD = 500;
    private static final int DEFAULT_SILENCE_CHUNKS_BEFORE_RESET = 20;
    private static final int DEFAULT_SOFT_RESET_INTERVAL_CHUNKS = 50;
    private static final String PREFS_NAME = "anju_kws";
    private static final String PREF_MODEL_ID = "selected_model_id";
    private static final String MODEL_REGISTRY_ASSET = "model_registry.json";
    private static final String ROOT_MODEL_FILE = "kws.onnx";
    private static final String ROOT_CONFIG_FILE = "kws_runtime_config.json";
    private static final Pattern SCORE_PATTERN = Pattern.compile("score=([0-9.]+)");
    private static final Pattern LATENCY_PATTERN = Pattern.compile("infer_ms=([0-9.]+)");
    private static final String KEYWORD_TEXT = "\u5b89\u5c45\u5c0f\u5b9d";
    private static final int WAKEUP_TONE_DURATION_MS = 160;
    private static final int WAKEUP_TONE_VOLUME = 100;
    private static final boolean DEFAULT_ENABLE_WAKEUP_TONE = true;
    private static final int DEFAULT_SLIDING_WINDOW_MS = 1200;
    private static final int DEFAULT_SLIDING_HOP_MS = 100;
    private static final int DEFAULT_SLIDING_CONSECUTIVE_HITS = 2;
    private static final int DEFAULT_SLIDING_COOLDOWN_MS = 1800;

    private volatile boolean startRecord = false;
    private AudioRecord record = null;
    private int recorderBufferSize = 0;
    private final BlockingQueue<short[]> bufferQueue = new ArrayBlockingQueue<>(MAX_QUEUE_SIZE);
    private float threshold = DEFAULT_THRESHOLD;
    private int wakeupCount = 0;
    private long lastWakeupUiTimeMs = 0;
    private int audioDebugChunkCount = 0;
    private long sessionStartMs = 0;
    private String sessionId = "";
    private File sessionLogFile = null;
    private File sessionPcmFile = null;
    private File sessionRawPcmFile = null;
    private File sessionEnhancedPcmFile = null;
    private final Object sessionLogLock = new Object();
    private final StringBuilder sessionLog = new StringBuilder();
    private String lastNativeDebug = "";
    private ToneGenerator wakeupToneGenerator;
    private short[] slidingRingBuffer = new short[1];
    private int slidingRingWrite = 0;
    private int slidingRingCount = 0;
    private int slidingSamplesSinceScore = 0;
    private int slidingConsecutiveHits = 0;
    private int slidingWindowIndex = 0;
    private long slidingTotalSamples = 0;
    private long lastSlidingTriggerMs = 0;

    private LinearLayout statusPanel;
    private TextView statusText;
    private TextView keywordText;
    private TextView detailText;
    private TextView scoreText;
    private TextView latencyText;
    private TextView thresholdText;
    private TextView eventCountText;
    private TextView modelText;
    private Spinner modelSpinner;
    private Button listenToggleButton;
    private Button button;
    private VoiceRectView voiceView;
    private final List<ModelOption> modelOptions = new ArrayList<>();
    private String selectedModelId = "";
    private StreamingConfig streamingConfig = StreamingConfig.defaults();

    private static class StreamingConfig {
        final float speechRmsThreshold;
        final int speechPeakThreshold;
        final int silenceChunksBeforeReset;
        final int softResetIntervalChunks;
        final int slidingWindowSamples;
        final int slidingHopSamples;
        final int slidingConsecutiveHits;
        final int slidingCooldownMs;
        final boolean enableWakeupTone;

        StreamingConfig(float speechRmsThreshold, int speechPeakThreshold,
                        int silenceChunksBeforeReset, int softResetIntervalChunks,
                        int slidingWindowSamples, int slidingHopSamples,
                        int slidingConsecutiveHits, int slidingCooldownMs,
                        boolean enableWakeupTone) {
            this.speechRmsThreshold = speechRmsThreshold;
            this.speechPeakThreshold = speechPeakThreshold;
            this.silenceChunksBeforeReset = silenceChunksBeforeReset;
            this.softResetIntervalChunks = softResetIntervalChunks;
            this.slidingWindowSamples = slidingWindowSamples;
            this.slidingHopSamples = slidingHopSamples;
            this.slidingConsecutiveHits = slidingConsecutiveHits;
            this.slidingCooldownMs = slidingCooldownMs;
            this.enableWakeupTone = enableWakeupTone;
        }

        static StreamingConfig defaults() {
            return new StreamingConfig(
                    DEFAULT_SPEECH_RMS_THRESHOLD,
                    DEFAULT_SPEECH_PEAK_THRESHOLD,
                    DEFAULT_SILENCE_CHUNKS_BEFORE_RESET,
                    DEFAULT_SOFT_RESET_INTERVAL_CHUNKS,
                    msToSamples(DEFAULT_SLIDING_WINDOW_MS),
                    msToSamples(DEFAULT_SLIDING_HOP_MS),
                    DEFAULT_SLIDING_CONSECUTIVE_HITS,
                    DEFAULT_SLIDING_COOLDOWN_MS,
                    DEFAULT_ENABLE_WAKEUP_TONE);
        }
    }

    private static class ModelOption {
        final String id;
        final String displayName;
        final String modelAsset;
        final String configAsset;

        ModelOption(String id, String displayName, String modelAsset, String configAsset) {
            this.id = id;
            this.displayName = displayName;
            this.modelAsset = modelAsset;
            this.configAsset = configAsset;
        }

        @Override
        public String toString() {
            return displayName;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        bindViews();
        wakeupToneGenerator = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, WAKEUP_TONE_VOLUME);
        updateUiForIdle();
        requestAudioPermissions();
        try {
            loadModelRegistry();
            applySelectedModel(false);
            setupModelSpinner();
        } catch (IOException e) {
            Log.e(LOG_TAG, "Error processing asset files", e);
            Toast.makeText(this, "Model asset init failed", Toast.LENGTH_LONG).show();
        }

        listenToggleButton.setOnClickListener(view -> {
            if (!startRecord) {
                startRecording();
            } else {
                stopRecording();
            }
        });
        button.setOnClickListener(view -> resetStats());
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions,
                                           int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_PERMISSIONS_RECORD_AUDIO) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i(LOG_TAG, "record permission is granted");
                initRecorder();
            } else {
                Toast.makeText(this, "Record audio permission denied", Toast.LENGTH_LONG).show();
                listenToggleButton.setEnabled(false);
                detailText.setText("Microphone permission denied");
            }
        }
    }

    private void bindViews() {
        statusPanel = findViewById(R.id.statusPanel);
        statusText = findViewById(R.id.statusText);
        keywordText = findViewById(R.id.keywordText);
        detailText = findViewById(R.id.detailText);
        scoreText = findViewById(R.id.scoreText);
        latencyText = findViewById(R.id.latencyText);
        thresholdText = findViewById(R.id.thresholdText);
        eventCountText = findViewById(R.id.eventCountText);
        modelText = findViewById(R.id.modelText);
        modelSpinner = findViewById(R.id.modelSpinner);
        listenToggleButton = findViewById(R.id.listenToggleButton);
        button = findViewById(R.id.button);
        voiceView = findViewById(R.id.voiceRectView);
        keywordText.setText(KEYWORD_TEXT);
    }

    private void loadModelRegistry() throws IOException {
        modelOptions.clear();
        AssetManager assetMgr = getAssets();
        byte[] data;
        try (InputStream is = assetMgr.open(MODEL_REGISTRY_ASSET)) {
            data = readAllBytes(is);
        }
        try {
            JSONObject json = new JSONObject(new String(data, StandardCharsets.UTF_8));
            String defaultModelId = json.optString("default_model_id", "");
            JSONArray models = json.getJSONArray("models");
            for (int i = 0; i < models.length(); i++) {
                JSONObject model = models.getJSONObject(i);
                modelOptions.add(new ModelOption(
                        model.getString("id"),
                        model.getString("display_name"),
                        model.getString("model_asset"),
                        model.getString("config_asset")));
            }
            SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
            selectedModelId = prefs.getString(PREF_MODEL_ID, defaultModelId);
        } catch (JSONException e) {
            throw new IOException("invalid model registry", e);
        }
        if (modelOptions.isEmpty()) {
            modelOptions.add(new ModelOption("default", "Default", ROOT_MODEL_FILE, ROOT_CONFIG_FILE));
            selectedModelId = "default";
        }
        if (findModelIndex(selectedModelId) < 0) {
            selectedModelId = modelOptions.get(0).id;
        }
    }

    private byte[] readAllBytes(InputStream is) throws IOException {
        java.io.ByteArrayOutputStream output = new java.io.ByteArrayOutputStream();
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
            output.write(buffer, 0, read);
        }
        return output.toByteArray();
    }

    private void setupModelSpinner() {
        ArrayAdapter<ModelOption> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, modelOptions);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelSpinner.setAdapter(adapter);
        int selectedIndex = Math.max(0, findModelIndex(selectedModelId));
        modelSpinner.setSelection(selectedIndex, false);
        modelSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, android.view.View view,
                                       int position, long id) {
                ModelOption option = modelOptions.get(position);
                if (option.id.equals(selectedModelId)) {
                    return;
                }
                selectedModelId = option.id;
                getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
                        .edit()
                        .putString(PREF_MODEL_ID, selectedModelId)
                        .apply();
                try {
                    applySelectedModel(true);
                } catch (IOException e) {
                    Log.e(LOG_TAG, "Model switch failed", e);
                    Toast.makeText(MainActivity.this, "Model switch failed", Toast.LENGTH_LONG).show();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });
    }

    private int findModelIndex(String modelId) {
        for (int i = 0; i < modelOptions.size(); i++) {
            if (modelOptions.get(i).id.equals(modelId)) {
                return i;
            }
        }
        return -1;
    }

    private ModelOption selectedModel() {
        int index = findModelIndex(selectedModelId);
        return modelOptions.get(index >= 0 ? index : 0);
    }

    private void applySelectedModel(boolean fromUser) throws IOException {
        if (startRecord) {
            stopRecording();
        }
        ModelOption option = selectedModel();
        copyAsset(option.modelAsset, new File(getFilesDir(), ROOT_MODEL_FILE));
        copyAsset(option.configAsset, new File(getFilesDir(), ROOT_CONFIG_FILE));
        JSONObject runtimeConfig = readRuntimeConfig();
        threshold = readThreshold(runtimeConfig);
        streamingConfig = readStreamingConfig(runtimeConfig);
        Spot.init(getFilesDir().getPath(), threshold);
        Spot.configureStreaming(streamingConfig.speechRmsThreshold,
                streamingConfig.speechPeakThreshold,
                streamingConfig.silenceChunksBeforeReset,
                streamingConfig.softResetIntervalChunks);
        Spot.reset();
        thresholdText.setText(String.format(Locale.US, "Threshold\n%.3f", threshold));
        modelText.setText("Model: " + option.displayName);
        if (fromUser) {
            resetStats();
            Toast.makeText(this, "Switched model: " + option.displayName, Toast.LENGTH_SHORT).show();
        }
        Log.i(LOG_TAG, "active_model id=" + option.id + " name=" + option.displayName
                + " threshold=" + threshold
                + " speech_rms_threshold=" + streamingConfig.speechRmsThreshold
                + " speech_peak_threshold=" + streamingConfig.speechPeakThreshold
                + " silence_chunks_before_reset=" + streamingConfig.silenceChunksBeforeReset
                + " soft_reset_interval_chunks=" + streamingConfig.softResetIntervalChunks
                + " sliding_window_samples=" + streamingConfig.slidingWindowSamples
                + " sliding_hop_samples=" + streamingConfig.slidingHopSamples
                + " sliding_consecutive_hits=" + streamingConfig.slidingConsecutiveHits
                + " sliding_cooldown_ms=" + streamingConfig.slidingCooldownMs
                + " enable_wakeup_tone=" + streamingConfig.enableWakeupTone);
    }

    private void copyAsset(String assetPath, File dst) throws IOException {
        Log.i(LOG_TAG, "Copying " + assetPath + " to " + dst.getAbsolutePath());
        try (InputStream is = getAssets().open(assetPath);
             OutputStream os = new FileOutputStream(dst, false)) {
            byte[] buffer = new byte[16 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }
    }

    private JSONObject readRuntimeConfig() {
        File config = new File(getFilesDir(), "kws_runtime_config.json");
        try (InputStream is = new java.io.FileInputStream(config)) {
            byte[] data = new byte[(int) config.length()];
            int read = is.read(data);
            if (read <= 0) {
                return new JSONObject();
            }
            return new JSONObject(new String(data, 0, read));
        } catch (IOException | JSONException e) {
            Log.w(LOG_TAG, "Use default runtime config", e);
            return new JSONObject();
        }
    }

    private float readThreshold(JSONObject json) {
        return (float) json.optDouble("threshold_initial", DEFAULT_THRESHOLD);
    }

    private StreamingConfig readStreamingConfig(JSONObject json) {
        return new StreamingConfig(
                (float) json.optDouble("speech_rms_threshold", DEFAULT_SPEECH_RMS_THRESHOLD),
                json.optInt("speech_peak_threshold", DEFAULT_SPEECH_PEAK_THRESHOLD),
                json.optInt("silence_chunks_before_reset", DEFAULT_SILENCE_CHUNKS_BEFORE_RESET),
                json.optInt("soft_reset_interval_chunks", DEFAULT_SOFT_RESET_INTERVAL_CHUNKS),
                msToSamples(json.optInt("sliding_window_ms", DEFAULT_SLIDING_WINDOW_MS)),
                msToSamples(json.optInt("sliding_hop_ms", DEFAULT_SLIDING_HOP_MS)),
                Math.max(1, json.optInt("sliding_consecutive_hits",
                        DEFAULT_SLIDING_CONSECUTIVE_HITS)),
                Math.max(0, json.optInt("sliding_cooldown_ms",
                        DEFAULT_SLIDING_COOLDOWN_MS)),
                json.optBoolean("enable_wakeup_tone", DEFAULT_ENABLE_WAKEUP_TONE));
    }

    private static int msToSamples(int milliseconds) {
        return Math.max(READ_CHUNK_SAMPLES,
                Math.round(SAMPLE_RATE * Math.max(1, milliseconds) / 1000.0f));
    }

    private void requestAudioPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    MY_PERMISSIONS_RECORD_AUDIO);
        } else {
            initRecorder();
        }
    }

    private void initRecorder() {
        int minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT);
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(LOG_TAG, "Audio buffer can't initialize");
            detailText.setText("Audio buffer init failed");
            return;
        }
        recorderBufferSize = Math.max(minBufferSize, READ_CHUNK_SAMPLES * 2 * 4);
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            return;
        }
        record = new AudioRecord(MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                recorderBufferSize);
        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "AudioRecord can't initialize");
            detailText.setText("AudioRecord init failed");
            return;
        }
        Log.i(LOG_TAG, "Record init okay, bufferBytes=" + recorderBufferSize);
    }

    private void startRecording() {
        if (record == null) {
            initRecorder();
        }
        if (record == null || startRecord) {
            return;
        }
        startRecord = true;
        bufferQueue.clear();
        audioDebugChunkCount = 0;
        lastNativeDebug = "";
        wakeupCount = 0;
        lastWakeupUiTimeMs = 0;
        resetSlidingState();
        eventCountText.setText(String.format(Locale.US, "Wakeups\n%d", wakeupCount));
        beginSession();
        Spot.reset();
        updateUiForListening("Sliding-window audio");
        startRecordThread();
        startSlidingSpotThread();
        modelSpinner.setEnabled(false);
        listenToggleButton.setText("停止监听");
    }

    private void stopRecording() {
        if (!startRecord) {
            return;
        }
        startRecord = false;
        Spot.setInputFinished();
        appendSessionLog("session_stop_requested");
        saveSessionLogAfterThreadsStop();
        modelSpinner.setEnabled(true);
        listenToggleButton.setText("开始监听");
        updateUiForIdle();
    }

    private void startRecordThread() {
        new Thread(() -> {
            record.startRecording();
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
            File rawCaptureFile = sessionRawPcmFile != null
                    ? sessionRawPcmFile
                    : new File(getExternalFilesDir(null), "live_capture_raw_16k_s16le.pcm");
            try (FileOutputStream rawCapture = new FileOutputStream(rawCaptureFile, false)) {
                Log.i(LOG_TAG, "audio_raw_capture path=" + rawCaptureFile.getAbsolutePath());
                appendSessionLog("audio_raw_capture path=" + rawCaptureFile.getAbsolutePath());
                while (startRecord) {
                    short[] buffer = new short[READ_CHUNK_SAMPLES];
                    int read = record.read(buffer, 0, buffer.length);
                    if (read > 0) {
                        short[] data = read == buffer.length ? buffer : Arrays.copyOf(buffer, read);
                        AudioPreprocessor.Stats stats = computeRawStats(data);
                        double db = calculateDb(data);
                        logAudioDebug(read, stats);
                        writePcm(rawCapture, data);
                        runOnUiThread(() -> voiceView.add(db));
                        if (!bufferQueue.offer(data)) {
                            bufferQueue.poll();
                            bufferQueue.offer(data);
                        }
                    } else if (read < 0) {
                        Log.w(LOG_TAG, "audio_read_error read=" + read);
                        appendSessionLog("audio_read_error read=" + read);
                    }
                }
            } catch (IOException e) {
                Log.e(LOG_TAG, "audio_capture failed", e);
                appendSessionLog("audio_capture failed " + e.getMessage());
            }
            record.stop();
            appendSessionLog("audio_record_thread_stopped");
            runOnUiThread(() -> voiceView.zero());
        }, "anju-audio-record").start();
    }

    private double calculateDb(short[] buffer) {
        double energy = 0.0;
        for (short value : buffer) {
            energy += value * value;
        }
        energy /= Math.max(1, buffer.length);
        energy = (10 * Math.log10(1 + energy)) / 100;
        return Math.min(energy, 1.0);
    }

    private AudioPreprocessor.Stats computeRawStats(short[] buffer) {
        long sumSquares = 0;
        int peak = 0;
        for (short value : buffer) {
            int abs = Math.abs((int) value);
            if (abs > peak) {
                peak = abs;
            }
            sumSquares += (long) value * value;
        }
        double rms = Math.sqrt(sumSquares / (double) Math.max(1, buffer.length));
        AudioPreprocessor.Stats stats = new AudioPreprocessor.Stats();
        stats.rawRms = rms;
        stats.rawPeak = peak;
        stats.enhancedRms = rms;
        stats.enhancedPeak = peak;
        stats.gainDb = 0.0;
        stats.clippedCount = 0;
        stats.noiseFloorRms = 0.0;
        stats.speechDetected = false;
        stats.agcApplied = false;
        return stats;
    }

    private void logAudioDebug(int read, AudioPreprocessor.Stats stats) {
        audioDebugChunkCount++;
        if (audioDebugChunkCount % 25 != 0) {
            return;
        }
        String message = String.format(Locale.US,
                "audio_debug read=%d raw_rms=%.1f raw_peak=%d enhanced_rms=%.1f "
                        + "enhanced_peak=%d gain_db=%.1f clipped=%d noise_floor=%.1f "
                        + "speech=%s agc=%s queue=%d",
                read,
                stats.rawRms,
                stats.rawPeak,
                stats.enhancedRms,
                stats.enhancedPeak,
                stats.gainDb,
                stats.clippedCount,
                stats.noiseFloorRms,
                stats.speechDetected ? "true" : "false",
                stats.agcApplied ? "true" : "false",
                bufferQueue.size());
        Log.i(LOG_TAG, message);
        appendSessionLog(message);
    }

    private void writePcm(FileOutputStream output, short[] data) throws IOException {
        byte[] bytes = new byte[data.length * 2];
        for (int i = 0; i < data.length; i++) {
            bytes[i * 2] = (byte) (data[i] & 0xff);
            bytes[i * 2 + 1] = (byte) ((data[i] >> 8) & 0xff);
        }
        output.write(bytes);
    }

    private void resetSlidingState() {
        int windowSamples = Math.max(READ_CHUNK_SAMPLES, streamingConfig.slidingWindowSamples);
        slidingRingBuffer = new short[windowSamples];
        slidingRingWrite = 0;
        slidingRingCount = 0;
        slidingSamplesSinceScore = 0;
        slidingConsecutiveHits = 0;
        slidingWindowIndex = 0;
        slidingTotalSamples = 0;
        lastSlidingTriggerMs = 0;
    }

    private void appendSlidingSamples(short[] data) {
        for (short sample : data) {
            slidingRingBuffer[slidingRingWrite] = sample;
            slidingRingWrite = (slidingRingWrite + 1) % slidingRingBuffer.length;
            if (slidingRingCount < slidingRingBuffer.length) {
                slidingRingCount++;
            }
        }
        slidingTotalSamples += data.length;
        slidingSamplesSinceScore += data.length;
    }

    private short[] copySlidingWindow() {
        if (slidingRingCount < slidingRingBuffer.length) {
            return null;
        }
        short[] window = new short[slidingRingBuffer.length];
        int start = slidingRingWrite;
        for (int i = 0; i < slidingRingBuffer.length; i++) {
            window[i] = slidingRingBuffer[(start + i) % slidingRingBuffer.length];
        }
        return window;
    }

    private void startSlidingSpotThread() {
        new Thread(() -> {
            while (startRecord || bufferQueue.size() > 0) {
                try {
                    short[] data = bufferQueue.poll(100, TimeUnit.MILLISECONDS);
                    if (data == null) {
                        continue;
                    }
                    appendSlidingSamples(data);
                    while (slidingRingCount >= slidingRingBuffer.length
                            && slidingSamplesSinceScore >= streamingConfig.slidingHopSamples) {
                        slidingSamplesSinceScore -= streamingConfig.slidingHopSamples;
                        short[] window = copySlidingWindow();
                        if (window == null) {
                            continue;
                        }
                        int windowIndex = ++slidingWindowIndex;
                        long windowEndMs = slidingTotalSamples * 1000L / SAMPLE_RATE;
                        long windowStartMs = Math.max(0,
                                windowEndMs - slidingRingBuffer.length * 1000L / SAMPLE_RATE);
                        String result = Spot.scoreWindow(window);
                        if (result == null || result.length() == 0) {
                            continue;
                        }
                        float score = extractFloat(result, SCORE_PATTERN, 0.0f);
                        boolean windowHit = result.startsWith("WAKEUP");
                        if (windowHit) {
                            slidingConsecutiveHits++;
                        } else {
                            slidingConsecutiveHits = 0;
                        }
                        long now = System.currentTimeMillis();
                        boolean trigger = windowHit
                                && slidingConsecutiveHits >= streamingConfig.slidingConsecutiveHits
                                && now - lastSlidingTriggerMs >= streamingConfig.slidingCooldownMs;
                        if (trigger) {
                            lastSlidingTriggerMs = now;
                            slidingConsecutiveHits = 0;
                        }
                        String message = String.format(Locale.US,
                                "sliding_window index=%d window_start_ms=%d "
                                        + "window_end_ms=%d score=%.3f hit=%s "
                                        + "consecutive_hits=%d trigger=%s queue=%d result=\"%s\"",
                                windowIndex,
                                windowStartMs,
                                windowEndMs,
                                score,
                                windowHit ? "true" : "false",
                                slidingConsecutiveHits,
                                trigger ? "true" : "false",
                                bufferQueue.size(),
                                result);
                        Log.i(LOG_TAG, message);
                        appendSessionLog(message);
                        appendNativeDebugIfChanged();
                        boolean finalTrigger = trigger;
                        runOnUiThread(() -> updateUiFromSlidingResult(result, finalTrigger));
                    }
                } catch (InterruptedException e) {
                    Log.e(LOG_TAG, "sliding spot thread interrupted", e);
                    appendSessionLog("sliding_spot_thread_interrupted " + e.getMessage());
                    Thread.currentThread().interrupt();
                    return;
                }
            }
            appendSessionLog("sliding_spot_thread_stopped");
        }, "anju-kws-sliding-spot").start();
    }

    private void startAcceptWaveThread() {
        new Thread(() -> {
            while (startRecord || bufferQueue.size() > 0) {
                try {
                    short[] data = bufferQueue.poll(100, TimeUnit.MILLISECONDS);
                    if (data != null) {
                        Spot.acceptWaveform(data);
                    }
                } catch (InterruptedException e) {
                    Log.e(LOG_TAG, "accept thread interrupted", e);
                    appendSessionLog("accept_thread_interrupted " + e.getMessage());
                    Thread.currentThread().interrupt();
                    return;
                }
            }
            appendSessionLog("accept_thread_stopped");
        }, "anju-audio-accept").start();
    }

    private void startSpotThread() {
        new Thread(() -> {
            while (startRecord) {
                Spot.startSpot();
                String result = Spot.getResult();
                if (result == null || result.length() == 0) {
                    continue;
                }
                Log.i(LOG_TAG, result);
                appendSessionLog(result);
                appendNativeDebugIfChanged();
                runOnUiThread(() -> updateUiFromResult(result));
            }
            appendSessionLog("spot_thread_stopped");
        }, "anju-kws-spot").start();
    }

    private void updateUiFromResult(String result) {
        float score = extractFloat(result, SCORE_PATTERN, 0.0f);
        float latency = extractFloat(result, LATENCY_PATTERN, 0.0f);
        scoreText.setText(String.format(Locale.US, "Score\n%.3f", score));
        latencyText.setText(String.format(Locale.US, "Latency\n%.1f ms", latency));
        if (result.startsWith("WAKEUP")) {
            long now = System.currentTimeMillis();
            if (now - lastWakeupUiTimeMs > 800) {
                wakeupCount++;
                lastWakeupUiTimeMs = now;
                if (streamingConfig.enableWakeupTone) {
                    playWakeupTone(score);
                } else {
                    appendSessionLog(String.format(Locale.US,
                            "wakeup_tone_disabled score=%.3f", score));
                }
            }
            eventCountText.setText(String.format(Locale.US, "Wakeups\n%d", wakeupCount));
            updateUiForWakeup(score);
        } else {
            updateUiForListening(String.format(Locale.US, "Listening, score %.3f", score));
        }
    }

    private void updateUiFromSlidingResult(String result, boolean trigger) {
        float score = extractFloat(result, SCORE_PATTERN, 0.0f);
        float latency = extractFloat(result, LATENCY_PATTERN, 0.0f);
        scoreText.setText(String.format(Locale.US, "Score\n%.3f", score));
        latencyText.setText(String.format(Locale.US, "Latency\n%.1f ms", latency));
        if (trigger) {
            wakeupCount++;
            lastWakeupUiTimeMs = System.currentTimeMillis();
            eventCountText.setText(String.format(Locale.US, "Wakeups\n%d", wakeupCount));
            if (streamingConfig.enableWakeupTone) {
                playWakeupTone(score);
            } else {
                appendSessionLog(String.format(Locale.US,
                        "wakeup_tone_disabled score=%.3f", score));
            }
            updateUiForWakeup(score);
        } else {
            updateUiForListening(String.format(Locale.US, "Sliding, score %.3f", score));
        }
    }

    private float extractFloat(String text, Pattern pattern, float fallback) {
        Matcher matcher = pattern.matcher(text);
        if (!matcher.find()) {
            return fallback;
        }
        try {
            return Float.parseFloat(matcher.group(1));
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private void updateUiForIdle() {
        setPanelColor("#6B7280");
        statusText.setText("Idle");
        if (sessionLogFile != null) {
            detailText.setText("Log saved: " + sessionLogFile.getName());
        } else {
            detailText.setText("Recorder stopped");
        }
    }

    private void updateUiForListening(String detail) {
        setPanelColor("#2F80ED");
        statusText.setText("\u76d1\u542c\u4e2d");
        detailText.setText(detail);
    }

    private void updateUiForWakeup(float score) {
        setPanelColor("#1F9D55");
        statusText.setText("\u5524\u9192\u6210\u529f");
        detailText.setText(String.format(Locale.US, "%s detected, score %.3f", KEYWORD_TEXT, score));
    }

    private void playWakeupTone(float score) {
        if (wakeupToneGenerator == null) {
            return;
        }
        wakeupToneGenerator.stopTone();
        wakeupToneGenerator.startTone(ToneGenerator.TONE_PROP_BEEP, WAKEUP_TONE_DURATION_MS);
        appendSessionLog(String.format(Locale.US, "wakeup_tone_played type=single_beep score=%.3f", score));
    }

    private void setPanelColor(String color) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(Color.parseColor(color));
        drawable.setCornerRadius(28.0f);
        statusPanel.setBackground(drawable);
    }

    private void resetStats() {
        wakeupCount = 0;
        lastWakeupUiTimeMs = 0;
        eventCountText.setText(String.format(Locale.US, "Wakeups\n%d", wakeupCount));
        scoreText.setText("Score\n0.000");
        latencyText.setText("Latency\n0 ms");
        if (startRecord) {
            updateUiForListening("Sliding-window audio");
        } else {
            updateUiForIdle();
        }
    }

    private void beginSession() {
        sessionStartMs = System.currentTimeMillis();
        sessionId = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                .format(new Date(sessionStartMs));
        File root = getExternalFilesDir(null);
        File logDir = new File(root, "logs");
        File captureDir = new File(root, "captures");
        if (!logDir.exists() && !logDir.mkdirs()) {
            Log.w(LOG_TAG, "failed to create log dir: " + logDir.getAbsolutePath());
        }
        if (!captureDir.exists() && !captureDir.mkdirs()) {
            Log.w(LOG_TAG, "failed to create capture dir: " + captureDir.getAbsolutePath());
        }
        sessionLogFile = new File(logDir, "listen_session_" + sessionId + ".log");
        sessionRawPcmFile = new File(captureDir,
                "listen_session_" + sessionId + "_raw_16k_s16le.pcm");
        sessionEnhancedPcmFile = null;
        sessionPcmFile = sessionRawPcmFile;
        synchronized (sessionLogLock) {
            sessionLog.setLength(0);
        }
        appendSessionLog("session_start id=" + sessionId
                + " sample_rate=" + SAMPLE_RATE
                + " channels=1 format=s16le"
                + " threshold=" + threshold
                + " speech_rms_threshold=" + streamingConfig.speechRmsThreshold
                + " speech_peak_threshold=" + streamingConfig.speechPeakThreshold
                + " silence_chunks_before_reset=" + streamingConfig.silenceChunksBeforeReset
                + " soft_reset_interval_chunks=" + streamingConfig.softResetIntervalChunks
                + " detection_mode=sliding_window"
                + " sliding_window_samples=" + streamingConfig.slidingWindowSamples
                + " sliding_hop_samples=" + streamingConfig.slidingHopSamples
                + " sliding_consecutive_hits=" + streamingConfig.slidingConsecutiveHits
                + " sliding_cooldown_ms=" + streamingConfig.slidingCooldownMs
                + " enable_wakeup_tone=" + streamingConfig.enableWakeupTone
                + " raw_pcm=" + sessionRawPcmFile.getAbsolutePath()
                + " preprocessing=raw_only");
    }

    private void appendSessionLog(String message) {
        String time = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
                .format(new Date());
        synchronized (sessionLogLock) {
            sessionLog.append(time).append(' ').append(message).append('\n');
        }
    }

    private void appendNativeDebugIfChanged() {
        String debug = Spot.getDebug();
        if (debug == null || debug.length() == 0 || debug.equals(lastNativeDebug)) {
            return;
        }
        lastNativeDebug = debug;
        appendSessionLog("native_debug " + debug);
    }

    private void saveSessionLog() {
        if (sessionLogFile == null) {
            return;
        }
        long durationMs = Math.max(0, System.currentTimeMillis() - sessionStartMs);
        appendSessionLog("session_end id=" + sessionId
                + " duration_ms=" + durationMs
                + " wakeups=" + wakeupCount);
        String content;
        synchronized (sessionLogLock) {
            content = sessionLog.toString();
        }
        try (FileOutputStream output = new FileOutputStream(sessionLogFile, false)) {
            output.write(content.getBytes(StandardCharsets.UTF_8));
            output.flush();
            Log.i(LOG_TAG, "session_log_saved path=" + sessionLogFile.getAbsolutePath());
        } catch (IOException e) {
            Log.e(LOG_TAG, "session_log_save_failed", e);
        }
    }

    private void saveSessionLogAfterThreadsStop() {
        new Thread(() -> {
            try {
                Thread.sleep(700);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            saveSessionLog();
            runOnUiThread(this::updateUiForIdle);
        }, "anju-session-log-save").start();
    }

    @Override
    protected void onDestroy() {
        if (startRecord) {
            stopRecording();
        }
        if (wakeupToneGenerator != null) {
            wakeupToneGenerator.release();
            wakeupToneGenerator = null;
        }
        super.onDestroy();
    }
}

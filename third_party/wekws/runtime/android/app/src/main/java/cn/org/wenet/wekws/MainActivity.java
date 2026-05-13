package cn.org.wenet.wekws;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

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
    private static final List<String> RESOURCE = Arrays.asList("kws.onnx", "kws_runtime_config.json");
    private static final Pattern SCORE_PATTERN = Pattern.compile("score=([0-9.]+)");
    private static final Pattern LATENCY_PATTERN = Pattern.compile("infer_ms=([0-9.]+)");
    private static final String KEYWORD_TEXT = "\u5b89\u5c45\u5c0f\u5b9d";

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
    private final Object sessionLogLock = new Object();
    private final StringBuilder sessionLog = new StringBuilder();
    private String lastNativeDebug = "";

    private LinearLayout statusPanel;
    private TextView statusText;
    private TextView keywordText;
    private TextView detailText;
    private TextView scoreText;
    private TextView latencyText;
    private TextView thresholdText;
    private TextView eventCountText;
    private Button listenToggleButton;
    private Button button;
    private VoiceRectView voiceView;

    public static void assetsInit(Context context) throws IOException {
        AssetManager assetMgr = context.getAssets();
        for (String file : assetMgr.list("")) {
            if (RESOURCE.contains(file)) {
                File dst = new File(context.getFilesDir(), file);
                Log.i(LOG_TAG, "Copying " + file + " to " + dst.getAbsolutePath());
                try (InputStream is = assetMgr.open(file);
                     OutputStream os = new FileOutputStream(dst, false)) {
                    byte[] buffer = new byte[4 * 1024];
                    int read;
                    while ((read = is.read(buffer)) != -1) {
                        os.write(buffer, 0, read);
                    }
                    os.flush();
                }
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        bindViews();
        updateUiForIdle();
        requestAudioPermissions();
        try {
            assetsInit(this);
        } catch (IOException e) {
            Log.e(LOG_TAG, "Error processing asset files", e);
        }
        threshold = readThreshold();
        Spot.init(getFilesDir().getPath(), threshold);
        thresholdText.setText(String.format(Locale.US, "Threshold\n%.3f", threshold));

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
        listenToggleButton = findViewById(R.id.listenToggleButton);
        button = findViewById(R.id.button);
        voiceView = findViewById(R.id.voiceRectView);
        keywordText.setText(KEYWORD_TEXT);
    }

    private float readThreshold() {
        File config = new File(getFilesDir(), "kws_runtime_config.json");
        try (InputStream is = new java.io.FileInputStream(config)) {
            byte[] data = new byte[(int) config.length()];
            int read = is.read(data);
            if (read <= 0) {
                return DEFAULT_THRESHOLD;
            }
            JSONObject json = new JSONObject(new String(data, 0, read));
            return (float) json.optDouble("threshold_initial", DEFAULT_THRESHOLD);
        } catch (IOException | JSONException e) {
            Log.w(LOG_TAG, "Use default threshold: " + DEFAULT_THRESHOLD, e);
            return DEFAULT_THRESHOLD;
        }
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
        eventCountText.setText(String.format(Locale.US, "Wakeups\n%d", wakeupCount));
        beginSession();
        Spot.reset();
        updateUiForListening("Streaming audio");
        startRecordThread();
        startAcceptWaveThread();
        startSpotThread();
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
        listenToggleButton.setText("开始监听");
        updateUiForIdle();
    }

    private void startRecordThread() {
        new Thread(() -> {
            record.startRecording();
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
            File captureFile = sessionPcmFile != null
                    ? sessionPcmFile
                    : new File(getExternalFilesDir(null), "live_capture_16k_s16le.pcm");
            try (FileOutputStream capture = new FileOutputStream(captureFile, false)) {
                Log.i(LOG_TAG, "audio_capture path=" + captureFile.getAbsolutePath());
                appendSessionLog("audio_capture path=" + captureFile.getAbsolutePath());
                while (startRecord) {
                    short[] buffer = new short[READ_CHUNK_SAMPLES];
                    int read = record.read(buffer, 0, buffer.length);
                    if (read > 0) {
                        short[] data = read == buffer.length ? buffer : Arrays.copyOf(buffer, read);
                        double db = calculateDb(data);
                        logAudioDebug(data, read);
                        writePcm(capture, data);
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

    private void logAudioDebug(short[] buffer, int read) {
        audioDebugChunkCount++;
        if (audioDebugChunkCount % 25 != 0) {
            return;
        }
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
        String message = String.format(Locale.US,
                "audio_debug read=%d rms=%.1f peak=%d queue=%d",
                read, rms, peak, bufferQueue.size());
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
            }
            eventCountText.setText(String.format(Locale.US, "Wakeups\n%d", wakeupCount));
            updateUiForWakeup(score);
        } else {
            updateUiForListening(String.format(Locale.US, "Listening, score %.3f", score));
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
            updateUiForListening("Streaming audio");
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
        sessionPcmFile = new File(captureDir, "listen_session_" + sessionId + "_16k_s16le.pcm");
        synchronized (sessionLogLock) {
            sessionLog.setLength(0);
        }
        appendSessionLog("session_start id=" + sessionId
                + " sample_rate=" + SAMPLE_RATE
                + " channels=1 format=s16le"
                + " threshold=" + threshold
                + " pcm=" + sessionPcmFile.getAbsolutePath());
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
}

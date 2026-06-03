package cn.org.wenet.wekws;

import android.Manifest;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.IBinder;
import android.os.Process;
import android.util.Log;

import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Locale;

public class FarfieldCaptureService extends Service {
    private static final String LOG_TAG = "WEKWS-COLLECT";
    private static final int SAMPLE_RATE = 16000;
    private static final int READ_CHUNK_SAMPLES = 640;
    private static final int NOTIFICATION_ID = 2107;
    private static final String NOTIFICATION_CHANNEL = "farfield_capture";

    private final Object stateLock = new Object();
    private final StringBuilder sessionLog = new StringBuilder();

    private volatile boolean collecting = false;
    private String activeCaptureId = "";
    private String label = "";
    private String distanceM = "";
    private String sourceSampleId = "";
    private long startMs = 0;
    private CaptureContract.Files files = null;
    private AudioRecord recorder = null;
    private long rawSumSquares = 0L;
    private long totalSamples = 0L;
    private int rawPeak = 0;
    private long clippedSamples = 0L;

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        ensureForeground();
        String action = intent == null ? "" : intent.getAction();
        if (CaptureContract.ACTION_START.equals(action)) {
            startCapture(intent);
        } else if (CaptureContract.ACTION_STOP.equals(action)) {
            stopCapture(intent.getStringExtra(CaptureContract.EXTRA_CAPTURE_ID));
        } else if (CaptureContract.ACTION_DELETE.equals(action)) {
            deleteCapture(intent.getStringExtra(CaptureContract.EXTRA_CAPTURE_ID));
            stopIfIdle();
        } else if (CaptureContract.ACTION_RECOVER.equals(action)) {
            writeRecoveryIndex();
            stopIfIdle();
        } else if (CaptureContract.ACTION_STATUS.equals(action)) {
            writeActiveStatus();
            stopIfIdle();
        } else {
            Log.w(LOG_TAG, "unknown_action " + action);
            stopIfIdle();
        }
        return START_NOT_STICKY;
    }

    private void startCapture(Intent intent) {
        String captureId = intent.getStringExtra(CaptureContract.EXTRA_CAPTURE_ID);
        if (!CaptureContract.isValidCaptureId(captureId)) {
            Log.e(LOG_TAG, "invalid_capture_id " + captureId);
            stopIfIdle();
            return;
        }
        synchronized (stateLock) {
            if (collecting) {
                writeFailedStatus(captureId, "capture_already_in_progress " + activeCaptureId);
                return;
            }
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                writeFailedStatus(captureId, "record_audio_permission_missing");
                stopIfIdle();
                return;
            }
            files = CaptureContract.filesFor(captureId, getExternalFilesDir(null));
            if (!files.ensureDirectories()) {
                writeFailedStatus(captureId, "cannot_create_capture_directories");
                stopIfIdle();
                return;
            }
            if (CaptureContract.hasRecoverableStatus(files)) {
                Log.w(LOG_TAG, "capture_rejected_pending_recovery id=" + captureId);
                stopIfIdle();
                return;
            }
            activeCaptureId = captureId;
            label = valueOrEmpty(intent.getStringExtra(CaptureContract.EXTRA_LABEL));
            distanceM = valueOrEmpty(intent.getStringExtra(CaptureContract.EXTRA_DISTANCE_M));
            sourceSampleId = valueOrEmpty(intent.getStringExtra(CaptureContract.EXTRA_SOURCE_SAMPLE_ID));
            startMs = System.currentTimeMillis();
            rawSumSquares = 0L;
            totalSamples = 0L;
            rawPeak = 0;
            clippedSamples = 0L;
            sessionLog.setLength(0);
            appendLog("session_start id=" + activeCaptureId
                    + " label=" + label
                    + " distance_m=" + distanceM
                    + " source_sample_id=" + sourceSampleId
                    + " sample_rate=" + SAMPLE_RATE
                    + " channels=1 format=s16le"
                    + " capture_mode=raw_only");
            collecting = true;
            writeStatus("recording", "");
            new Thread(this::recordLoop, "anju-farfield-collect").start();
        }
    }

    private void stopCapture(String captureId) {
        synchronized (stateLock) {
            if (!collecting) {
                Log.w(LOG_TAG, "stop_without_active_capture id=" + captureId);
                stopIfIdle();
                return;
            }
            if (!activeCaptureId.equals(captureId)) {
                writeFailedStatus(captureId, "active_capture_mismatch " + activeCaptureId);
                return;
            }
            appendLog("session_stop_requested");
            collecting = false;
            writeStatus("stopping", "");
        }
    }

    private void recordLoop() {
        int minBufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (minBufferSize <= 0) {
            completeWithError("invalid_audio_buffer_size " + minBufferSize);
            return;
        }
        int bufferBytes = Math.max(minBufferSize, READ_CHUNK_SAMPLES * 2 * 4);
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            completeWithError("record_audio_permission_missing");
            return;
        }
        recorder = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferBytes);
        if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
            recorder.release();
            recorder = null;
            completeWithError("audio_record_init_failed");
            return;
        }
        try (FileOutputStream rawOutput = new FileOutputStream(files.rawPcm, false)) {
            recorder.startRecording();
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
            appendLog("audio_raw_capture path=" + files.rawPcm.getAbsolutePath());
            int chunkCount = 0;
            while (collecting) {
                short[] input = new short[READ_CHUNK_SAMPLES];
                int read = recorder.read(input, 0, input.length);
                if (read <= 0) {
                    appendLog("audio_read_error read=" + read);
                    continue;
                }
                short[] raw = read == input.length ? input : Arrays.copyOf(input, read);
                updateStats(raw);
                writePcm(rawOutput, raw);
                chunkCount++;
                if (chunkCount % 25 == 0) {
                    appendLog(String.format(Locale.US,
                            "audio_debug raw_rms=%.1f raw_peak=%d clipped=%d",
                            rms(rawSumSquares, totalSamples),
                            rawPeak,
                            clippedSamples));
                }
            }
            rawOutput.flush();
        } catch (IOException error) {
            completeWithError("audio_capture_io_failed " + error.getMessage());
            return;
        } finally {
            if (recorder != null) {
                try {
                    recorder.stop();
                } catch (IllegalStateException ignored) {
                    Log.w(LOG_TAG, "recorder_stop_failed", ignored);
                }
                recorder.release();
                recorder = null;
            }
        }
        appendLog("audio_record_thread_stopped");
        finishCapture("complete", "");
    }

    private void updateStats(short[] raw) {
        for (int index = 0; index < raw.length; index++) {
            int rawValue = raw[index];
            rawSumSquares += (long) rawValue * rawValue;
            rawPeak = Math.max(rawPeak, Math.abs(rawValue));
            if (Math.abs(rawValue) >= 30000) {
                clippedSamples++;
            }
        }
        totalSamples += raw.length;
    }

    private void completeWithError(String error) {
        appendLog("capture_failed " + error);
        collecting = false;
        finishCapture("failed", error);
    }

    private void finishCapture(String state, String error) {
        appendLog("session_end id=" + activeCaptureId
                + " duration_ms=" + Math.max(0L, System.currentTimeMillis() - startMs)
                + " state=" + state);
        saveLog();
        writeStatus(state, error);
        synchronized (stateLock) {
            collecting = false;
        }
        stopForeground(true);
        stopSelf();
    }

    private void deleteCapture(String captureId) {
        if (!CaptureContract.isValidCaptureId(captureId)) {
            Log.e(LOG_TAG, "delete_invalid_capture_id " + captureId);
            return;
        }
        synchronized (stateLock) {
            if (collecting && activeCaptureId.equals(captureId)) {
                Log.w(LOG_TAG, "delete_rejected_active_capture " + captureId);
                return;
            }
        }
        CaptureContract.Files target = CaptureContract.filesFor(captureId, getExternalFilesDir(null));
        deleteFile(target.rawPcm);
        deleteFile(target.enhancedPcm);
        deleteFile(target.log);
        deleteFile(target.status);
        Log.i(LOG_TAG, "capture_deleted id=" + captureId);
    }

    private void writeRecoveryIndex() {
        File statusDir = new File(getExternalFilesDir(null), "status");
        JSONArray entries = new JSONArray();
        File[] statusFiles = statusDir.listFiles();
        if (statusFiles != null) {
            for (File statusFile : statusFiles) {
                if (statusFile.getName().endsWith(".json")
                        && !"recover.json".equals(statusFile.getName())) {
                    entries.put(statusFile.getName());
                }
            }
        }
        JSONObject recovery = new JSONObject();
        try {
            recovery.put("state", "recovery_index");
            recovery.put("entries", entries);
        } catch (JSONException ignored) {
            Log.w(LOG_TAG, "recovery_json_failed", ignored);
        }
        writeJson(new File(statusDir, "recover.json"), recovery);
    }

    private void writeActiveStatus() {
        synchronized (stateLock) {
            if (files != null && CaptureContract.isValidCaptureId(activeCaptureId)) {
                writeStatus(collecting ? "recording" : "complete", "");
            }
        }
    }

    private void writeFailedStatus(String captureId, String error) {
        if (!CaptureContract.isValidCaptureId(captureId)) {
            Log.e(LOG_TAG, "cannot_write_failed_status invalid_capture_id=" + captureId);
            return;
        }
        CaptureContract.Files failed = CaptureContract.filesFor(captureId, getExternalFilesDir(null));
        failed.ensureDirectories();
        JSONObject status = baseStatus(captureId, "failed", error);
        writeJson(failed.status, status);
    }

    private void writeStatus(String state, String error) {
        if (files == null) {
            return;
        }
        JSONObject status = baseStatus(activeCaptureId, state, error);
        try {
            status.put("label", label);
            status.put("distance_m", distanceM);
            status.put("source_sample_id", sourceSampleId);
            status.put("capture_mode", "raw_only");
            status.put("raw_pcm", files.rawPcm.getAbsolutePath());
            status.put("log", files.log.getAbsolutePath());
            status.put("samples", totalSamples);
            status.put("duration_sec", totalSamples / (double) SAMPLE_RATE);
            status.put("raw_rms", rms(rawSumSquares, totalSamples));
            status.put("raw_peak", rawPeak);
            status.put("clipping_ratio", totalSamples == 0 ? 0.0 : clippedSamples / (double) totalSamples);
        } catch (JSONException errorJson) {
            Log.e(LOG_TAG, "status_json_failed", errorJson);
        }
        writeJson(files.status, status);
    }

    private JSONObject baseStatus(String captureId, String state, String error) {
        JSONObject status = new JSONObject();
        try {
            status.put("capture_id", captureId);
            status.put("state", state);
            status.put("error", error);
            status.put("updated_at_ms", System.currentTimeMillis());
        } catch (JSONException ignored) {
            Log.w(LOG_TAG, "base_status_json_failed", ignored);
        }
        return status;
    }

    private double rms(long sumSquares, long sampleCount) {
        return sampleCount == 0 ? 0.0 : Math.sqrt(sumSquares / (double) sampleCount);
    }

    private void appendLog(String message) {
        String time = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US).format(new Date());
        synchronized (sessionLog) {
            sessionLog.append(time).append(' ').append(message).append('\n');
        }
        Log.i(LOG_TAG, message);
    }

    private void saveLog() {
        if (files == null) {
            return;
        }
        String content;
        synchronized (sessionLog) {
            content = sessionLog.toString();
        }
        try (FileOutputStream output = new FileOutputStream(files.log, false)) {
            output.write(content.getBytes(StandardCharsets.UTF_8));
            output.flush();
        } catch (IOException error) {
            Log.e(LOG_TAG, "log_save_failed", error);
        }
    }

    private void writeJson(File destination, JSONObject value) {
        File parent = destination.getParentFile();
        if (!parent.exists() && !parent.mkdirs()) {
            Log.e(LOG_TAG, "status_dir_create_failed " + parent.getAbsolutePath());
            return;
        }
        File temporary = new File(parent, destination.getName() + ".tmp");
        try (FileOutputStream output = new FileOutputStream(temporary, false)) {
            output.write(value.toString().getBytes(StandardCharsets.UTF_8));
            output.flush();
        } catch (IOException error) {
            Log.e(LOG_TAG, "status_write_failed", error);
            return;
        }
        if (!temporary.renameTo(destination)) {
            Log.e(LOG_TAG, "status_rename_failed " + destination.getAbsolutePath());
        }
    }

    private void writePcm(FileOutputStream output, short[] data) throws IOException {
        byte[] bytes = new byte[data.length * 2];
        for (int index = 0; index < data.length; index++) {
            bytes[index * 2] = (byte) (data[index] & 0xff);
            bytes[index * 2 + 1] = (byte) ((data[index] >> 8) & 0xff);
        }
        output.write(bytes);
    }

    private void deleteFile(File file) {
        if (file.exists() && !file.delete()) {
            Log.w(LOG_TAG, "delete_failed " + file.getAbsolutePath());
        }
    }

    private void stopIfIdle() {
        if (!collecting) {
            stopForeground(true);
            stopSelf();
        }
    }

    private String valueOrEmpty(String value) {
        return value == null ? "" : value;
    }

    private void ensureForeground() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(new NotificationChannel(
                    NOTIFICATION_CHANNEL, "Far-field capture", NotificationManager.IMPORTANCE_LOW));
        }
        Intent launchIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this,
                0,
                launchIntent,
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.M ? PendingIntent.FLAG_IMMUTABLE : 0);
        Notification notification = new NotificationCompat.Builder(this, NOTIFICATION_CHANNEL)
                .setContentTitle("AnJu KWS capture")
                .setContentText("Collecting physical far-field audio")
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentIntent(pendingIntent)
                .setOngoing(true)
                .build();
        startForeground(NOTIFICATION_ID, notification);
    }
}

package cn.org.wenet.wekws;

import java.io.File;
import java.util.regex.Pattern;

public final class CaptureContract {
    public static final String ACTION_START = "cn.org.wenet.wekws.action.COLLECT_START";
    public static final String ACTION_STOP = "cn.org.wenet.wekws.action.COLLECT_STOP";
    public static final String ACTION_STATUS = "cn.org.wenet.wekws.action.COLLECT_STATUS";
    public static final String ACTION_DELETE = "cn.org.wenet.wekws.action.COLLECT_DELETE";
    public static final String ACTION_RECOVER = "cn.org.wenet.wekws.action.COLLECT_RECOVER";

    public static final String EXTRA_CAPTURE_ID = "capture_id";
    public static final String EXTRA_LABEL = "label";
    public static final String EXTRA_DISTANCE_M = "distance_m";
    public static final String EXTRA_SOURCE_SAMPLE_ID = "source_sample_id";

    private static final Pattern SAFE_CAPTURE_ID = Pattern.compile("[A-Za-z0-9_-]{1,160}");

    private CaptureContract() {
    }

    public static boolean isValidCaptureId(String captureId) {
        return captureId != null && SAFE_CAPTURE_ID.matcher(captureId).matches();
    }

    public static Files filesFor(String captureId, File externalRoot) {
        if (!isValidCaptureId(captureId)) {
            throw new IllegalArgumentException("Invalid capture_id: " + captureId);
        }
        File root = externalRoot == null ? new File("") : externalRoot;
        File captureDir = new File(root, "captures");
        File logDir = new File(root, "logs");
        File statusDir = new File(root, "status");
        return new Files(
                new File(captureDir, captureId + "_raw_16k_s16le.pcm"),
                new File(captureDir, captureId + "_enhanced_16k_s16le.pcm"),
                new File(logDir, captureId + ".log"),
                new File(statusDir, captureId + ".json"));
    }

    public static boolean hasRecoverableStatus(Files files) {
        return files != null && files.status.exists();
    }

    public static final class Files {
        public final File rawPcm;
        public final File enhancedPcm;
        public final File log;
        public final File status;

        Files(File rawPcm, File enhancedPcm, File log, File status) {
            this.rawPcm = rawPcm;
            this.enhancedPcm = enhancedPcm;
            this.log = log;
            this.status = status;
        }

        public boolean ensureDirectories() {
            return ensure(rawPcm.getParentFile())
                    && ensure(log.getParentFile())
                    && ensure(status.getParentFile());
        }

        private boolean ensure(File directory) {
            return directory.exists() || directory.mkdirs();
        }
    }
}

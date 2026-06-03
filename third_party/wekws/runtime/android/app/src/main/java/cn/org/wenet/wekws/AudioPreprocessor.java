package cn.org.wenet.wekws;

public class AudioPreprocessor {
    public static class Stats {
        public double rawRms;
        public int rawPeak;
        public double enhancedRms;
        public int enhancedPeak;
        public double gainDb;
        public int clippedCount;
        public double noiseFloorRms;
        public boolean speechDetected;
        public boolean agcApplied;
    }

    private static final int SAMPLE_RATE = 16000;
    private static final double HIGH_PASS_CUTOFF_HZ = 80.0;
    private static final boolean ENABLE_MILD_AGC = true;
    private static final boolean ENABLE_NOISE_GATE = false;
    private static final double TARGET_SPEECH_RMS = 2300.0;
    private static final double MIN_GAIN = 1.00;
    private static final double MAX_SPEECH_GAIN = 2.50;
    private static final double MAX_NOISE_GAIN = 1.03;
    private static final double AGC_ATTACK = 0.08;
    private static final double AGC_RELEASE = 0.10;
    private static final int LIMITER_ABS = 30000;
    private static final double NOISE_GATE_ATTENUATION = 1.0;

    private final double highPassAlpha;
    private double previousInput = 0.0;
    private double previousHighPass = 0.0;
    private double smoothedGain = 1.0;
    private double noiseFloorRms = 280.0;
    private final Stats lastStats = new Stats();

    public AudioPreprocessor() {
        double rc = 1.0 / (2.0 * Math.PI * HIGH_PASS_CUTOFF_HZ);
        double dt = 1.0 / SAMPLE_RATE;
        highPassAlpha = rc / (rc + dt);
    }

    public void reset() {
        previousInput = 0.0;
        previousHighPass = 0.0;
        smoothedGain = 1.0;
        noiseFloorRms = 280.0;
    }

    public Stats getLastStats() {
        return lastStats;
    }

    public static String describe() {
        return "dc_remove,highpass_80hz,speech_aware_mild_agc,max_gain_8db,"
                + "noise_gate_off,limiter";
    }

    public short[] process(short[] input) {
        int length = input == null ? 0 : input.length;
        short[] output = new short[length];
        if (length == 0) {
            clearStats();
            return output;
        }

        RawStats raw = computeRawStats(input);
        updateNoiseFloor(raw.rms);
        boolean speech = isSpeech(raw.rms, raw.peak);

        double[] filtered = new double[length];
        double sumSquares = 0.0;
        double mean = raw.mean;
        for (int i = 0; i < length; i++) {
            double centered = input[i] - mean;
            double highPassed = highPassAlpha * (previousHighPass + centered - previousInput);
            previousInput = centered;
            previousHighPass = highPassed;
            filtered[i] = highPassed;
            sumSquares += highPassed * highPassed;
        }

        double filteredRms = Math.sqrt(sumSquares / Math.max(1, length));
        boolean agcApplied = ENABLE_MILD_AGC && isAgcEligible(raw.rms, raw.peak, speech);
        double desiredGain = 1.0;
        if (agcApplied && filteredRms > 1.0) {
            desiredGain = TARGET_SPEECH_RMS / filteredRms;
            desiredGain = clamp(desiredGain, MIN_GAIN, MAX_SPEECH_GAIN);
        }

        double smoothing = desiredGain > smoothedGain ? AGC_ATTACK : AGC_RELEASE;
        smoothedGain += smoothing * (desiredGain - smoothedGain);
        double outputGain = agcApplied ? smoothedGain : Math.min(smoothedGain, MAX_NOISE_GAIN);
        if (ENABLE_NOISE_GATE && !agcApplied) {
            outputGain *= NOISE_GATE_ATTENUATION;
        }

        long enhancedSumSquares = 0;
        int enhancedPeak = 0;
        int clipped = 0;
        for (int i = 0; i < length; i++) {
            int value = (int) Math.round(filtered[i] * outputGain);
            if (value > LIMITER_ABS) {
                value = LIMITER_ABS;
                clipped++;
            } else if (value < -LIMITER_ABS) {
                value = -LIMITER_ABS;
                clipped++;
            }
            output[i] = (short) value;
            int abs = Math.abs(value);
            if (abs > enhancedPeak) {
                enhancedPeak = abs;
            }
            enhancedSumSquares += (long) value * value;
        }

        lastStats.rawRms = raw.rms;
        lastStats.rawPeak = raw.peak;
        lastStats.enhancedRms = Math.sqrt(enhancedSumSquares / (double) Math.max(1, length));
        lastStats.enhancedPeak = enhancedPeak;
        lastStats.gainDb = 20.0 * Math.log10(Math.max(0.0001, outputGain));
        lastStats.clippedCount = clipped;
        lastStats.noiseFloorRms = noiseFloorRms;
        lastStats.speechDetected = speech;
        lastStats.agcApplied = agcApplied;
        return output;
    }

    private void clearStats() {
        lastStats.rawRms = 0.0;
        lastStats.rawPeak = 0;
        lastStats.enhancedRms = 0.0;
        lastStats.enhancedPeak = 0;
        lastStats.gainDb = 0.0;
        lastStats.clippedCount = 0;
        lastStats.noiseFloorRms = noiseFloorRms;
        lastStats.speechDetected = false;
        lastStats.agcApplied = false;
    }

    private RawStats computeRawStats(short[] input) {
        long sum = 0;
        long sumSquares = 0;
        int peak = 0;
        for (short sample : input) {
            int value = sample;
            int abs = Math.abs(value);
            if (abs > peak) {
                peak = abs;
            }
            sum += value;
            sumSquares += (long) value * value;
        }
        double mean = sum / (double) Math.max(1, input.length);
        double rms = Math.sqrt(sumSquares / (double) Math.max(1, input.length));
        return new RawStats(mean, rms, peak);
    }

    private void updateNoiseFloor(double rms) {
        if (rms <= 0.0) {
            return;
        }
        if (rms < noiseFloorRms * 1.8 || rms < 650.0) {
            noiseFloorRms = 0.98 * noiseFloorRms + 0.02 * rms;
        } else {
            noiseFloorRms = 0.999 * noiseFloorRms + 0.001 * rms;
        }
        noiseFloorRms = clamp(noiseFloorRms, 80.0, 3000.0);
    }

    private boolean isSpeech(double rms, int peak) {
        double rmsThreshold = Math.max(300.0, noiseFloorRms * 2.2);
        double peakThreshold = Math.max(1400.0, noiseFloorRms * 4.5);
        return rms >= rmsThreshold || peak >= peakThreshold;
    }

    private boolean isAgcEligible(double rms, int peak, boolean speech) {
        if (speech) {
            return true;
        }
        double rmsThreshold = Math.max(450.0, noiseFloorRms * 1.45);
        double peakThreshold = Math.max(1800.0, noiseFloorRms * 3.2);
        return rms >= rmsThreshold || peak >= peakThreshold;
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private static class RawStats {
        final double mean;
        final double rms;
        final int peak;

        RawStats(double mean, double rms, int peak) {
            this.mean = mean;
            this.rms = rms;
            this.peak = peak;
        }
    }
}

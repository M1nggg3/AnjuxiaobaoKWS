package cn.org.wenet.wekws;

public class Spot {

    static {
        System.loadLibrary("wekws");
    }

    public static native void init(String modelDir, float threshold);
    public static native void reset();
    public static native void setThreshold(float threshold);
    public static native void acceptWaveform(short[] waveform);
    public static native void setInputFinished();
    public static native void startSpot();
    public static native String getResult();
    public static native String getDebug();
}

package com.nvel.app;

public class Llm {
    static {
        System.loadLibrary("nvel_llm");
    }

    public interface Sink {
        void onBytes(byte[] chunk);
    }

    public static native void nativeInit(String nativeLibDir);

    /** Returns null on success, otherwise an error message. */
    public static native String nativeLoad(String path);

    public static native String nativeInfo();

    public static native void nativeCancel();

    public static native void nativeFree();

    /** Single user message. Returns null on success, otherwise an error message. */
    public static native String nativeGenerate(String prompt, int maxTokens, float temp, Sink sink);

    /**
     * Multi-turn chat. messagesJson is a JSON array of
     * {"role":"system|user|assistant","content":"..."}.
     * Returns null on success, otherwise an error message.
     */
    public static native String nativeChat(String messagesJson, int maxTokens, float temp, Sink sink);
}

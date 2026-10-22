package com.nvel.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.webkit.JavascriptInterface;
import android.webkit.JsResult;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public class MainActivity extends Activity {
    private static final int REQ_PICK_MODEL = 1001;
    private static final int REQ_EXPORT = 1002;
    private static final String MODEL_NAME = "model.gguf";

    private WebView webView;
    private File modelFile;
    private boolean nativeReady = false;
    private boolean modelLoaded = false;
    private volatile boolean busy = false;
    private String busyWhat = "";

    // pending export payload (written after user picks save location)
    private String exportName;
    private String exportMime;
    private String exportText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(Color.parseColor("#06080b"));
        getWindow().setNavigationBarColor(Color.parseColor("#06080b"));
        modelFile = new File(getFilesDir(), MODEL_NAME);

        webView = new WebView(this);
        webView.setBackgroundColor(Color.parseColor("#06080b"));
        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        // without a WebChromeClient the page's alert() / confirm() do nothing
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onJsAlert(WebView view, String url, String message, final JsResult result) {
                new AlertDialog.Builder(MainActivity.this)
                        .setMessage(message)
                        .setPositiveButton("OK", (d, w) -> result.confirm())
                        .setOnCancelListener(d -> result.cancel())
                        .show();
                return true;
            }

            @Override
            public boolean onJsConfirm(WebView view, String url, String message, final JsResult result) {
                new AlertDialog.Builder(MainActivity.this)
                        .setMessage(message)
                        .setPositiveButton("OK", (d, w) -> result.confirm())
                        .setNegativeButton("Отмена", (d, w) -> result.cancel())
                        .setOnCancelListener(d -> result.cancel())
                        .show();
                return true;
            }
        });
        webView.addJavascriptInterface(new Bridge(), "NvelAI");
        setContentView(webView);
        webView.loadUrl("file:///android_asset/index.html");
    }

    @Override
    public void onBackPressed() {
        // let the page handle chat navigation first
        webView.evaluateJavascript(
                "(function(){ try { return window.nvelBack && window.nvelBack(); } catch(e){ return false; } })()",
                value -> {
                    if ("true".equals(value)) return;
                    if (webView.canGoBack()) {
                        webView.goBack();
                    } else {
                        MainActivity.super.onBackPressed();
                    }
                });
    }

    @Override
    protected void onDestroy() {
        if (nativeReady) {
            try {
                Llm.nativeCancel();
            } catch (Throwable ignored) {
            }
        }
        super.onDestroy();
    }

    // ---------- events to the page ----------

    private void emit(JSONObject o) {
        final String js = "window.nvelAiEvent && window.nvelAiEvent(" + o.toString() + ");";
        webView.post(() -> webView.evaluateJavascript(js, null));
    }

    private void emit(String type, String key, Object value) {
        try {
            JSONObject o = new JSONObject();
            o.put("type", type);
            if (key != null) o.put(key, value);
            emit(o);
        } catch (Exception ignored) {
        }
    }

    private synchronized boolean tryBegin(String what) {
        if (busy) return false;
        busy = true;
        busyWhat = what;
        return true;
    }

    private synchronized void end() {
        busy = false;
        busyWhat = "";
    }

    // ---------- model file picker / export ----------

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_EXPORT) {
            if (resultCode != RESULT_OK || data == null || data.getData() == null) {
                exportName = exportMime = exportText = null;
                emit("exportCancel", null, null);
                return;
            }
            final Uri uri = data.getData();
            final String text = exportText;
            exportName = exportMime = exportText = null;
            new Thread(() -> {
                try {
                    OutputStream out = getContentResolver().openOutputStream(uri);
                    if (out == null) {
                        emit("exportError", "message", "Не удалось открыть файл для записи");
                        return;
                    }
                    try {
                        out.write(text.getBytes(StandardCharsets.UTF_8));
                    } finally {
                        out.close();
                    }
                    emit("exportDone", null, null);
                } catch (Throwable t) {
                    emit("exportError", "message", "Ошибка сохранения: " + t.getMessage());
                }
            }, "nvel-export").start();
            return;
        }

        if (requestCode != REQ_PICK_MODEL) return;
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            emit("copyCancel", null, null);
            return;
        }
        final Uri uri = data.getData();
        if (!tryBegin("copy")) {
            emit("error", "message", "Сейчас идёт другая операция");
            return;
        }
        new Thread(() -> {
            try {
                copyModel(uri);
            } finally {
                end();
            }
        }, "nvel-copy").start();
    }

    private void copyModel(Uri uri) {
        File tmp = new File(getFilesDir(), MODEL_NAME + ".part");
        try {
            long total = -1;
            Cursor c = getContentResolver().query(uri, null, null, null, null);
            if (c != null) {
                try {
                    int i = c.getColumnIndex(OpenableColumns.SIZE);
                    if (i >= 0 && c.moveToFirst() && !c.isNull(i)) total = c.getLong(i);
                } finally {
                    c.close();
                }
            }
            if (total > 0 && getFilesDir().getUsableSpace() < total + 200L * 1024 * 1024) {
                emit("error", "message", "Не хватает места на телефоне: нужно около "
                        + (total / (1024 * 1024)) + " МБ свободно");
                return;
            }

            InputStream in = getContentResolver().openInputStream(uri);
            if (in == null) {
                emit("error", "message", "Не удалось открыть файл");
                return;
            }
            try {
                byte[] magic = new byte[4];
                int got = 0;
                while (got < 4) {
                    int r = in.read(magic, got, 4 - got);
                    if (r < 0) break;
                    got += r;
                }
                if (got < 4 || magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
                    emit("error", "message", "Это не файл модели .gguf");
                    return;
                }

                if (modelLoaded) {
                    Llm.nativeFree();
                    modelLoaded = false;
                }

                OutputStream out = new FileOutputStream(tmp);
                try {
                    out.write(magic);
                    long done = 4;
                    long lastEmit = 0;
                    byte[] buf = new byte[1 << 20];
                    int n;
                    while ((n = in.read(buf)) > 0) {
                        out.write(buf, 0, n);
                        done += n;
                        long now = System.currentTimeMillis();
                        if (now - lastEmit > 300) {
                            lastEmit = now;
                            JSONObject o = new JSONObject();
                            o.put("type", "copyProgress");
                            o.put("done", done);
                            o.put("total", total);
                            emit(o);
                        }
                    }
                } finally {
                    out.close();
                }
            } finally {
                in.close();
            }

            if (modelFile.exists()) modelFile.delete();
            if (!tmp.renameTo(modelFile)) {
                emit("error", "message", "Не удалось сохранить модель");
                return;
            }
            emit("copyDone", null, null);
        } catch (Throwable t) {
            emit("error", "message", "Ошибка копирования: " + t.getMessage());
        } finally {
            if (tmp.exists()) tmp.delete();
        }
    }

    // ---------- generation ----------

    private void ensureNative() {
        if (!nativeReady) {
            Llm.nativeInit(getApplicationInfo().nativeLibraryDir);
            nativeReady = true;
        }
    }

    private boolean ensureModel() {
        if (!modelLoaded) {
            emit("loading", null, null);
            String err = Llm.nativeLoad(modelFile.getAbsolutePath());
            if (err != null) {
                emit("error", "message", err);
                return false;
            }
            modelLoaded = true;
        }
        return true;
    }

    private void runGenerate(final String prompt, final int maxTokens, final float temp) {
        try {
            ensureNative();
            if (!ensureModel()) return;
            emit("started", null, null);

            final ByteArrayOutputStream acc = new ByteArrayOutputStream();
            final long[] last = {0};
            String err = Llm.nativeGenerate(prompt, maxTokens, temp, chunk -> {
                acc.write(chunk, 0, chunk.length);
                long now = System.currentTimeMillis();
                if (now - last[0] > 100) {
                    last[0] = now;
                    emit("token", "text", decode(acc));
                }
            });
            if (err != null) {
                emit("error", "message", err);
            } else {
                emit("done", "text", decode(acc));
            }
        } catch (UnsatisfiedLinkError e) {
            emit("error", "message", "Нейросеть не поддерживается на этом телефоне (нужен 64-битный ARM)");
        } catch (Throwable t) {
            emit("error", "message", "Ошибка: " + t.getMessage());
        }
    }

    private void runChat(final String messagesJson, final int maxTokens, final float temp) {
        try {
            ensureNative();
            if (!ensureModel()) return;
            emit("started", null, null);

            final ByteArrayOutputStream acc = new ByteArrayOutputStream();
            final long[] last = {0};
            String err = Llm.nativeChat(messagesJson, maxTokens, temp, chunk -> {
                acc.write(chunk, 0, chunk.length);
                long now = System.currentTimeMillis();
                if (now - last[0] > 100) {
                    last[0] = now;
                    emit("token", "text", decode(acc));
                }
            });
            if (err != null) {
                // if we already streamed some text, still finish with done so UI keeps partial reply
                String partial = decode(acc);
                if (!partial.isEmpty()) {
                    emit("token", "text", partial);
                }
                emit("error", "message", err);
            } else {
                emit("done", "text", decode(acc));
            }
        } catch (UnsatisfiedLinkError e) {
            emit("error", "message", "Нейросеть не поддерживается на этом телефоне (нужен 64-битный ARM)");
        } catch (Throwable t) {
            emit("error", "message", "Ошибка: " + t.getMessage());
        }
    }

    private static String decode(ByteArrayOutputStream acc) {
        String s = new String(acc.toByteArray(), StandardCharsets.UTF_8);
        while (s.endsWith("�")) s = s.substring(0, s.length() - 1);
        return s;
    }

    // ---------- bridge for JavaScript ----------

    private class Bridge {
        @JavascriptInterface
        public String status() {
            try {
                JSONObject o = new JSONObject();
                o.put("available", true);
                o.put("hasModel", modelFile.exists());
                o.put("modelMb", modelFile.exists() ? modelFile.length() / (1024 * 1024) : 0);
                o.put("loaded", modelLoaded);
                o.put("busy", busy);
                o.put("busyWhat", busyWhat);
                return o.toString();
            } catch (Exception e) {
                return "{\"available\":false}";
            }
        }

        @JavascriptInterface
        public void pickModel() {
            runOnUiThread(() -> {
                if (busy) {
                    emit("error", "message", "Сейчас идёт другая операция");
                    return;
                }
                Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType("*/*");
                startActivityForResult(i, REQ_PICK_MODEL);
            });
        }

        @JavascriptInterface
        public void generate(final String prompt, final int maxTokens) {
            if (!modelFile.exists()) {
                emit("error", "message", "Сначала выбери файл модели");
                return;
            }
            if (!tryBegin("generate")) {
                emit("error", "message", "Нейросеть уже работает");
                return;
            }
            new Thread(() -> {
                try {
                    runGenerate(prompt, maxTokens, 0.3f);
                } finally {
                    end();
                }
            }, "nvel-llm").start();
        }

        @JavascriptInterface
        public void chat(final String messagesJson, final int maxTokens, final float temp) {
            if (!modelFile.exists()) {
                emit("error", "message", "Сначала выбери файл модели");
                return;
            }
            if (!tryBegin("generate")) {
                emit("error", "message", "Нейросеть уже работает");
                return;
            }
            final float t = temp > 0 ? temp : 0.7f;
            final int mt = maxTokens > 0 ? maxTokens : 512;
            new Thread(() -> {
                try {
                    runChat(messagesJson, mt, t);
                } finally {
                    end();
                }
            }, "nvel-llm").start();
        }

        @JavascriptInterface
        public void cancel() {
            if (nativeReady) {
                try {
                    Llm.nativeCancel();
                } catch (Throwable ignored) {
                }
            }
        }

        @JavascriptInterface
        public void deleteModel() {
            if (busy) {
                emit("error", "message", "Сейчас идёт другая операция");
                return;
            }
            try {
                if (modelLoaded) {
                    Llm.nativeFree();
                    modelLoaded = false;
                }
            } catch (Throwable ignored) {
            }
            if (modelFile.exists()) modelFile.delete();
            emit("modelDeleted", null, null);
        }

        @JavascriptInterface
        public void exportFile(final String name, final String mime, final String text) {
            runOnUiThread(() -> {
                exportName = name != null ? name : "nvel_export.txt";
                exportMime = mime != null ? mime : "text/plain";
                exportText = text != null ? text : "";
                Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
                i.addCategory(Intent.CATEGORY_OPENABLE);
                i.setType(exportMime);
                i.putExtra(Intent.EXTRA_TITLE, exportName);
                startActivityForResult(i, REQ_EXPORT);
            });
        }
    }
}

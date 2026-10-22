// Local LLM generation for NVEL: single prompt or multi-turn chat history.
#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include <unistd.h>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#ifdef __ANDROID__
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NVEL", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NVEL", __VA_ARGS__)
#else
#define LOGI(...) (fprintf(stderr, __VA_ARGS__), fprintf(stderr, "\n"))
#define LOGE(...) (fprintf(stderr, __VA_ARGS__), fprintf(stderr, "\n"))
#endif

namespace {

constexpr int CTX_SIZE = 4096;
constexpr int BATCH_SIZE = 512;

llama_model * g_model = nullptr;
std::atomic<bool> g_cancel(false);

void log_cb(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        LOGE("%s", text);
    } else {
        LOGI("%s", text);
    }
}

int pick_threads() {
    const int n = (int) sysconf(_SC_NPROCESSORS_ONLN);
    return std::max(2, std::min(4, n - 2));
}

jstring make_error(JNIEnv * env, const std::string & msg) {
    LOGE("%s", msg.c_str());
    return env->NewStringUTF(msg.c_str());
}

bool decode_prompt(llama_context * ctx, common_batch & batch, const std::vector<llama_token> & tokens) {
    const int n = (int) tokens.size();
    for (int i = 0; i < n; i += BATCH_SIZE) {
        const int len = std::min(BATCH_SIZE, n - i);
        batch.clear();
        for (int j = 0; j < len; j++) {
            batch.add(tokens[i + j], i + j, 0, i + j == n - 1);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            return false;
        }
    }
    return true;
}

// Minimal JSON array parser for [{"role":"...","content":"..."}, ...]
bool parse_messages_json(const std::string & json, std::vector<common_chat_msg> & out, std::string & err) {
    out.clear();
    size_t i = 0;
    auto skip_ws = [&]() {
        while (i < json.size() && std::isspace((unsigned char) json[i])) i++;
    };
    auto match = [&](char c) -> bool {
        skip_ws();
        if (i < json.size() && json[i] == c) { i++; return true; }
        return false;
    };
    auto parse_string = [&](std::string & s) -> bool {
        skip_ws();
        if (i >= json.size() || json[i] != '"') return false;
        i++;
        s.clear();
        while (i < json.size()) {
            char c = json[i++];
            if (c == '"') return true;
            if (c == '\\' && i < json.size()) {
                char e = json[i++];
                switch (e) {
                    case '"': case '\\': case '/': s.push_back(e); break;
                    case 'n': s.push_back('\n'); break;
                    case 'r': s.push_back('\r'); break;
                    case 't': s.push_back('\t'); break;
                    case 'b': s.push_back('\b'); break;
                    case 'f': s.push_back('\f'); break;
                    case 'u':
                        for (int k = 0; k < 4 && i < json.size(); k++) i++;
                        s.push_back('?');
                        break;
                    default: s.push_back(e); break;
                }
            } else {
                s.push_back(c);
            }
        }
        return false;
    };

    if (!match('[')) {
        err = "Ожидался JSON-массив сообщений";
        return false;
    }
    skip_ws();
    if (match(']')) return true;

    while (i < json.size()) {
        if (!match('{')) {
            err = "Ожидался объект сообщения";
            return false;
        }
        std::string role, content;
        bool have_role = false, have_content = false;
        while (true) {
            skip_ws();
            if (match('}')) break;
            std::string key;
            if (!parse_string(key)) {
                err = "Ошибка ключа JSON";
                return false;
            }
            if (!match(':')) {
                err = "Ожидалось ':'";
                return false;
            }
            skip_ws();
            if (i < json.size() && json[i] == '"') {
                std::string val;
                if (!parse_string(val)) {
                    err = "Ошибка значения JSON";
                    return false;
                }
                if (key == "role") { role = val; have_role = true; }
                else if (key == "content") { content = val; have_content = true; }
            } else if (i < json.size() && (json[i] == '{' || json[i] == '[')) {
                int depth = 0;
                char open = json[i], close = (open == '{') ? '}' : ']';
                do {
                    char c = json[i++];
                    if (c == open) depth++;
                    else if (c == close) depth--;
                    else if (c == '"') {
                        while (i < json.size()) {
                            if (json[i] == '\\') { i += 2; continue; }
                            if (json[i++] == '"') break;
                        }
                    }
                } while (depth > 0 && i < json.size());
            } else {
                while (i < json.size() && json[i] != ',' && json[i] != '}' && json[i] != ']') i++;
            }
            skip_ws();
            if (i < json.size() && json[i] == ',') { i++; continue; }
        }
        if (!have_role || !have_content) {
            err = "У сообщения нужны role и content";
            return false;
        }
        if (role != "system" && role != "user" && role != "assistant") {
            err = "Неизвестная роль: " + role;
            return false;
        }
        common_chat_msg msg;
        msg.role = role;
        msg.content = content;
        out.push_back(msg);

        skip_ws();
        if (match(']')) return true;
        if (!match(',')) {
            err = "Ожидалась запятая или ]";
            return false;
        }
    }
    err = "Обрезанный JSON";
    return false;
}

jstring run_generation(JNIEnv * env, const std::vector<common_chat_msg> & messages,
                       jint max_tokens, jfloat temp, jobject sink) {
    if (!g_model) {
        return make_error(env, "Модель не загружена");
    }
    g_cancel = false;

    jclass sink_cls = env->GetObjectClass(sink);
    jmethodID on_bytes = env->GetMethodID(sink_cls, "onBytes", "([B)V");
    if (!on_bytes) {
        return make_error(env, "Внутренняя ошибка: нет метода onBytes");
    }

    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;
    std::string error;

    try {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = CTX_SIZE;
        cp.n_batch = BATCH_SIZE;
        cp.n_ubatch = BATCH_SIZE;
        cp.n_threads = pick_threads();
        cp.n_threads_batch = cp.n_threads;
        ctx = llama_init_from_model(g_model, cp);
        if (!ctx) {
            return make_error(env, "Не хватает памяти для запуска модели");
        }

        common_chat_templates_ptr tmpls = common_chat_templates_init(g_model, "");
        common_chat_templates_inputs in;
        in.messages = messages;
        in.add_generation_prompt = true;
        in.use_jinja = true;
        in.enable_thinking = false;
        const std::string text = common_chat_templates_apply(tmpls.get(), in).prompt.str();

        std::vector<llama_token> tokens = common_tokenize(ctx, text, /*add_special=*/true, /*parse_special=*/true);
        if ((int) tokens.size() + max_tokens >= CTX_SIZE) {
            error = "Запрос слишком длинный для контекста модели";
        }

        if (error.empty()) {
            common_params_sampling sp;
            sp.temp = temp;
            smpl = common_sampler_init(g_model, sp);

            common_batch batch(ctx);
            if (!decode_prompt(ctx, batch, tokens)) {
                error = "Ошибка при обработке запроса";
            } else {
                const llama_vocab * vocab = llama_model_get_vocab(g_model);
                int pos = (int) tokens.size();
                for (int i = 0; i < max_tokens && !g_cancel; i++) {
                    const llama_token tok = common_sampler_sample(smpl, ctx, -1);
                    common_sampler_accept(smpl, tok, true);
                    if (llama_vocab_is_eog(vocab, tok)) {
                        break;
                    }

                    const std::string piece = common_token_to_piece(ctx, tok);
                    if (!piece.empty()) {
                        jbyteArray arr = env->NewByteArray((jsize) piece.size());
                        env->SetByteArrayRegion(arr, 0, (jsize) piece.size(), (const jbyte *) piece.data());
                        env->CallVoidMethod(sink, on_bytes, arr);
                        env->DeleteLocalRef(arr);
                        if (env->ExceptionCheck()) {
                            env->ExceptionClear();
                            break;
                        }
                    }

                    batch.clear();
                    batch.add(tok, pos++, 0, true);
                    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                        error = "Ошибка при генерации";
                        break;
                    }
                }
            }
        }
    } catch (const std::exception & e) {
        error = std::string("Ошибка: ") + e.what();
    }

    if (smpl) {
        common_sampler_free(smpl);
    }
    if (ctx) {
        llama_free(ctx);
    }
    if (!error.empty()) {
        return make_error(env, error);
    }
    return nullptr;
}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_nvel_app_Llm_nativeInit(JNIEnv * env, jclass, jstring jlibdir) {
    llama_log_set(log_cb, nullptr);
    const char * dir = env->GetStringUTFChars(jlibdir, nullptr);
    LOGI("Loading backends from %s", dir);
    ggml_backend_load_all_from_path(dir);
    env->ReleaseStringUTFChars(jlibdir, dir);
    llama_backend_init();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvel_app_Llm_nativeLoad(JNIEnv * env, jclass, jstring jpath) {
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    const char * path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!model) {
        return make_error(env, "Не удалось загрузить модель (файл повреждён, не хватает памяти или формат не поддерживается)");
    }
    g_model = model;
    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvel_app_Llm_nativeInfo(JNIEnv * env, jclass) {
    if (!g_model) {
        return env->NewStringUTF("");
    }
    char desc[128];
    llama_model_desc(g_model, desc, sizeof(desc));
    std::string s = std::string(desc) + " | " + llama_print_system_info();
    return env->NewStringUTF(s.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvel_app_Llm_nativeCancel(JNIEnv *, jclass) {
    g_cancel = true;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvel_app_Llm_nativeFree(JNIEnv *, jclass) {
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvel_app_Llm_nativeGenerate(JNIEnv * env, jclass, jstring jprompt, jint max_tokens, jfloat temp, jobject sink) {
    const char * prompt_c = env->GetStringUTFChars(jprompt, nullptr);
    const std::string prompt(prompt_c);
    env->ReleaseStringUTFChars(jprompt, prompt_c);

    std::vector<common_chat_msg> messages;
    common_chat_msg msg;
    msg.role = "user";
    msg.content = prompt;
    messages.push_back(msg);
    return run_generation(env, messages, max_tokens, temp, sink);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nvel_app_Llm_nativeChat(JNIEnv * env, jclass, jstring jmessages, jint max_tokens, jfloat temp, jobject sink) {
    const char * raw = env->GetStringUTFChars(jmessages, nullptr);
    const std::string json(raw);
    env->ReleaseStringUTFChars(jmessages, raw);

    std::vector<common_chat_msg> messages;
    std::string err;
    if (!parse_messages_json(json, messages, err)) {
        return make_error(env, err);
    }
    if (messages.empty()) {
        return make_error(env, "Нет сообщений");
    }
    return run_generation(env, messages, max_tokens, temp, sink);
}

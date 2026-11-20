#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <map>

// 名字和 TAG 只含中文/大写字母/数字，这里只是不信任输入地做一次转义
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// fill the batch with tokens at consecutive positions starting from pos_0, output logits only for the last one
static void batch_set_tokens(llama_batch_ext * batch, const llama_token * tokens, int32_t n_tokens, llama_pos pos_0) {
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const int32_t   idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = pos_0 + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, n_tokens - 1, true);
}

// 往 batch 里加一个 token：指定序列、位置，并要求输出它的 logits
static int32_t batch_add_one(llama_batch_ext * batch, llama_seq_id seq, llama_token tok, llama_pos pos, bool out_logits) {
    const int32_t idx = llama_batch_ext_add_token(batch, seq, tok);
    if (idx < 0) {
        return idx;
    }
    if (!llama_batch_ext_set_pos(batch, idx, &pos)) {
        return -1;
    }
    if (!llama_batch_ext_set_output_logits(batch, idx, out_logits)) {
        return -1;
    }
    return idx;
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

struct chat_message {
    std::string role;
    std::string content;
};

static std::string format_chat(const llama_model * model, const std::vector<chat_message> & history) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        throw std::runtime_error("The model has no default chat template");
    }

    std::vector<llama_chat_message> messages;
    messages.reserve(history.size());
    for (const auto & message : history) {
        messages.push_back({ message.role.c_str(), message.content.c_str() });
    }

    const int32_t required = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, nullptr, 0);
    if (required <= 0) {
        throw std::runtime_error("Chat template is unsupported or produced an empty prompt");
    }

    std::vector<char> buffer(required);
    const int32_t     written =
        llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, buffer.data(), required);
    if (written != required) {
        throw std::runtime_error("Chat template formatting failed");
    }
    return std::string(buffer.data(), written);
}

// 一条候选路线
struct beam {
    std::string              text;          // 对应的文字
    double                   score;         // 每步 log 概率之和
    llama_sampler *          grmr;          // 每条路自己的语法状态
    llama_seq_id             seq;           // 它在 KV cache 里的编号
    llama_pos                next_pos;      // 下一个 token 该放的位置（= 提示词长度 + 已写 token 数）
    const float*             logits;        // 指向最近一次算出的分数（下次 decode 前生效）
};


int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::vector<std::string> words;

    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt    = "Hello my name is";
    // number of layers to offload to the GPU
    int         ngl       = 99;
    // number of tokens to predict
    int         n_predict = 32;
    // number of beam to select token
    int n_beam = 4;
    // 常驻模式：从 stdin 逐行读中文名，每行回一条 JSON；模型、词表、前缀只准备一次
    bool serve = false;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-k") == 0) {
                n_beam = std::stoi(argv[++i]);
            } else if (strcmp(argv[i], "--serve") == 0) {
                serve = true;
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        for(; i < argc; ++i) {
            words.push_back(argv[i]);
        }
        if (!serve && words.empty()) {
            print_usage(argc, argv);
            return 1;
        }
    }

    llama_backend_init();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 把命令行输入的中文当作 user 消息，套上聊天模板
    const std::string  sys = "把中文测试场景名翻译成英文 TAG：只用大写字母和数字，不超过 16 个字符。只输出 TAG。";

    // 提示词 = 固定的 system + 中文名，公共前缀就是 system 那一段
    auto tokenize_prompt = [&](const std::string & w, std::vector<llama_token> & out) -> bool {
        const std::string p = format_chat(model, {{"system", sys}, {"user", w}});
        const int n = -llama_tokenize(vocab, p.c_str(), p.size(), NULL, 0, true, true);
        if (n <= 0) {
            return false;
        }
        out.resize(n);
        return llama_tokenize(vocab, p.c_str(), p.size(), out.data(), out.size(), true, true) >= 0;
    };

    std::vector<std::vector<llama_token>> prompts;
    std::vector<llama_token>             prefix;   // 解码进 seq 0、所有名字共用的开头

    if (serve) {
        // 两个不同的占位名字求出公共前缀，之后每个名字只解码自己的尾段
        std::vector<llama_token> a, b;
        if (!tokenize_prompt("甲", a) || !tokenize_prompt("乙乙乙", b)) {
            fprintf(stderr, "%s: error: failed to tokenize\n", __func__);
            return 1;
        }
        size_t i = 0;
        while (i < a.size() && i < b.size() && a[i] == b[i]) {
            ++i;
        }
        prefix.assign(a.begin(), a.begin() + i);
    } else {
        for (const auto & w : words) {
            std::vector<llama_token> t;
            if (!tokenize_prompt(w, t)) {
                fprintf(stderr, "%s: error: failed to tokenize\n", __func__);
                return 1;
            }
            prompts.push_back(std::move(t));
        }

        // 所有词的提示词在分词后共有的开头长度（不能靠字符串切分，分词可能跨边界合并）
        size_t common = prompts[0].size();
        for (const auto & t : prompts) {
            size_t i = 0;
            while (i < common && i < t.size() && t[i] == prompts[0][i]) {
                ++i;
            }
            common = i;
        }
        if (prompts.size() == 1) {
            common = 0;     // 只有一个词时没有可复用的前缀，整段算
        }
        prefix.assign(prompts[0].begin(), prompts[0].begin() + common);
    }
    const size_t common = prefix.size();
    fprintf(stderr, "%s: common prefix = %zu tokens, %zu prompts\n", __func__, common, prompts.size());

    size_t max_len = common + 64;   // 常驻模式下来一个名字现算，按最短可用来估
    for (const auto & t : prompts) {
        max_len = std::max(max_len, t.size());
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_seq_max  = 2 * n_beam + 1;
    ctx_params.kv_unified = true;
    ctx_params.n_ctx      = static_cast<uint32_t>(max_len + (2 * n_beam + 1) * n_predict + 256);
    ctx_params.n_batch    = static_cast<int>(std::max(max_len, static_cast<size_t>(n_beam)));
    ctx_params.no_perf    = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams         = llama_sampler_chain_default_params();
    sparams.no_perf      = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_sampler * grmr = llama_sampler_init_grammar(vocab, "root ::= [A-Z] [A-Z0-9]{0, 15}", "root");
    if (grmr == nullptr) {
        fprintf(stderr, "%s: error: failed to init grmr\n", __func__);
        return 1;
    }

    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    const auto t_main_start = ggml_time_us();
    int        n_decode     = 0;

    llama_memory_t mem = llama_get_memory(ctx);

    // 序列编号 0 留给公共前缀，1 .. 2k 轮流给各条路用
    std::vector<int> free_ids;
    for (int i = 1; i <= 2 * n_beam; ++i) {
        free_ids.push_back(i);
    }

    struct cand {
        double      score;   // 走到这一步的总分
        int         parent;  // 从 live 里第几条路延申出来
        llama_token tok;     // 这一步选的 token
    };

    const int n_vocab = llama_vocab_n_tokens(vocab);
    // 能出现在 TAG 里的
    std::vector<llama_token> tag_vocab;
    for (llama_token id = 0; id < n_vocab; ++id) {
        if (llama_vocab_is_eog(vocab, id)) {
            tag_vocab.push_back(id);
            continue;
        }
        char buf[128];
        const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        bool ok_chars = (n > 0);
        for (int i = 0; i < n && ok_chars; ++i) {
            ok_chars = (buf[i] >= 'A' && buf[i] <= 'Z') || (buf[i] >= '0' && buf[i] <= '9');
        }
        if (ok_chars) {
            tag_vocab.push_back(id);
        }
    }
    fprintf(stderr, "%s: tag vocab = %zu / %d tokens\n", __func__, tag_vocab.size(), n_vocab);

    // 词表约束做成 sampler：集合之外的 token 直接掩成 -inf，和语法约束串在一条链上
    llama_sampler * subset = llama_sampler_init_vocab_subset(vocab, tag_vocab.data(), tag_vocab.size());

    // 每条 beam 每一步都要重建候选，只在外面分配一次
    std::vector<llama_token_data> cur(n_vocab);

    // 公共前缀只算一次放进 seq 0；每个名字把自己的独有部分接在它后面
    if (common > 0) {
        llama_batch_ext_clear(batch);
        batch_set_tokens(batch, prefix.data(), static_cast<int>(common), 0);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            fprintf(stderr, "%s: error: llama_process failed (prefix)\n", __func__);
            return 1;
        }
        n_decode += static_cast<int>(common);
    }

    // 跑一个名字：尾段接在公共前缀后面，做 beam search，返回按分数降序的候选
    auto run_word = [&](const std::vector<llama_token> & pt,
                        std::vector<std::pair<std::string, double>> & out) -> bool {
        // 提示词前 common 个 token 和 seq 0 里的前缀不一致时不能复用，整段重算
        const bool reuse = common > 0 && pt.size() >= common &&
            std::equal(prefix.begin(), prefix.end(), pt.begin());
        const size_t from = reuse ? common : 0;

        // 借一个序列，装「公共前缀（拷过来）+ 这个名字独有的 token」
        const llama_seq_id wid = free_ids.back();
        free_ids.pop_back();
        llama_memory_seq_rm(mem, wid, -1, -1);
        if (reuse) {
            llama_memory_seq_cp(mem, 0, wid, -1, -1);
        }

        llama_batch_ext_clear(batch);
        for (size_t i = from; i < pt.size(); ++i) {
            const int32_t idx = batch_add_one(batch, wid, pt[i], static_cast<llama_pos>(i), i + 1 == pt.size());
            if (idx < 0) {
                fprintf(stderr, "%s: error: batch is full\n", __func__);
                return false;
            }
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            fprintf(stderr, "%s: error: llama_process failed (word)\n", __func__);
            return false;
        }
        n_decode += static_cast<int>(pt.size() - from);

        // 起点：一条内容为空、就绪位置是这个词提示词末尾的路
        std::vector<beam> live;
        live.push_back({ "", 0.0, llama_sampler_clone(grmr), wid, static_cast<llama_pos>(pt.size()),
                         llama_get_logits_ith(ctx, -1) });
        std::map<std::string, double> finished;     // 文字 -> 最高分

        while (!live.empty()) {
            // 用每条路【上一轮算好的 logits】产生候选
            std::vector<cand> pool;
            for (int b = 0; b < static_cast<int>(live.size()); ++b) {
                const float * logits = live[b].logits;

                // 语法约束之前，先算每个 token 的 log 概率
                float max_l = logits[0];
                for (int i = 1; i < n_vocab; ++i) {
                    max_l = std::max(max_l, logits[i]);
                }
                double sum = 0.0;
                for (int i = 0; i < n_vocab; ++i) {
                    sum += std::exp(logits[i] - max_l);
                }
                const double log_sum = std::log(sum);

                // 先按词表掩码，再用这条路自己的语法状态筛掉违规 token
                for (int i = 0; i < n_vocab; ++i) {
                    cur[i] = { i, logits[i], 0.0f };
                }
                llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
                llama_sampler_apply(subset, &arr);
                llama_sampler_apply(live[b].grmr, &arr);

                // 允许的里面，按 lp 取前 k 个
                std::vector<std::pair<double, llama_token>> ok;
                for (int j = 0; j < static_cast<int>(arr.size); ++j) {
                    if (arr.data[j].logit != -INFINITY) {
                        const llama_token id = arr.data[j].id;
                        ok.push_back({ static_cast<double>(logits[id]) - max_l - log_sum, id });
                    }
                }
                const int n_ok = std::min<int>(n_beam, ok.size());
                std::partial_sort(ok.begin(), ok.begin() + n_ok, ok.end(),
                                  [](const auto & x, const auto & y) { return x.first > y.first; });
                for (int j = 0; j < n_ok; ++j) {
                    pool.push_back({ live[b].score + ok[j].first, b, ok[j].second });
                }
            }

            // 所有路的候选合在一起排名次
            std::sort(pool.begin(), pool.end(), [](const cand & x, const cand & y) { return x.score > y.score; });

            // 只给活下来的 K 个建新序列
            std::vector<beam> next;
            std::vector<int32_t> batch_ids;

            llama_batch_ext_clear(batch);
            for (int c = 0; c < std::min<int>(n_beam, pool.size()); ++c) {
                const beam & p = live[pool[c].parent];
                if (llama_vocab_is_eog(vocab, pool[c].tok)) {
                    // 进 finished（按文字去重留高分），不建新路、不占序列
                    auto it = finished.find(p.text);
                    if (it == finished.end() || it->second < pool[c].score) {
                        finished[p.text] = pool[c].score;
                    }
                    continue;
                }
                // 给子路分一个空闲序列，把父路的全部记忆挂过来
                const llama_seq_id id = free_ids.back();
                free_ids.pop_back();
                llama_memory_seq_rm(mem, id, -1, -1);
                llama_memory_seq_cp(mem, p.seq, id, -1, -1);

                // 只把这一步新选的 token 放进 batch，位置接在父路后面
                const int32_t idx = batch_add_one(batch, id, pool[c].tok, p.next_pos, true);
                if (idx < 0) {
                    fprintf(stderr, "%s: error: batch is full\n", __func__);
                    return 1;
                }
                batch_ids.push_back(idx);

                char buf[128];
                const int n = llama_token_to_piece(vocab, pool[c].tok, buf, sizeof(buf), 0, true);
                beam nb;
                nb.text = p.text + std::string(buf, n);
                nb.score = pool[c].score;
                nb.grmr = llama_sampler_clone(p.grmr);
                llama_sampler_accept(nb.grmr, pool[c].tok);
                nb.seq = id;
                nb.next_pos = p.next_pos + 1;
                nb.logits = nullptr;    // 下面算完再填
                next.push_back(nb);
            }

            // 所有子路的新 token 放在同一个 batch 里，一次算完
            if (!batch_ids.empty()) {
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                    fprintf(stderr, "%s: error: llama_process failed\n", __func__);
                    return 1;
                }
                n_decode += static_cast<int>(batch_ids.size());
                for (int j = 0; j < static_cast<int>(next.size()); ++j) {
                    next[j].logits = llama_get_logits_ith(ctx, batch_ids[j]);
                }
            }

            // 父路用完了：语法状态销毁，序列清空、编号还回池子（模板序列 0 一直保留）
            for (auto & b : live) {
                llama_sampler_free(b.grmr);
                if (b.seq != 0) {
                    llama_memory_seq_rm(mem, b.seq, -1, -1);
                    free_ids.push_back(b.seq);
                }
            }
            live = std::move(next);
        }

        out.assign(finished.begin(), finished.end());
        std::sort(out.begin(), out.end(), [](const auto & x, const auto & y) { return x.second > y.second; });
        if (out.size() > static_cast<size_t>(n_beam)) {
            out.resize(n_beam);
        }
        return true;
    };

    if (serve) {
        // 一行一个中文名，一行一条 JSON；调用方按行读，超时由调用方兜底
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            std::vector<llama_token> pt;
            std::vector<std::pair<std::string, double>> out;
            const char * err = nullptr;
            if (!tokenize_prompt(line, pt)) {
                err = "tokenize";
            } else if (!run_word(pt, out)) {
                err = "decode";
            }
            if (err) {
                printf("{\"name\":\"%s\",\"error\":\"%s\"}\n", json_escape(line).c_str(), err);
            } else {
                printf("{\"name\":\"%s\",\"candidates\":[", json_escape(line).c_str());
                for (size_t i = 0; i < out.size(); ++i) {
                    printf("%s{\"tag\":\"%s\",\"score\":%.4f}", i ? "," : "",
                           json_escape(out[i].first).c_str(), out[i].second);
                }
                printf("]}\n");
            }
            fflush(stdout);
        }
    } else {
        for (size_t w = 0; w < prompts.size(); ++w) {
            if (common > 0 && prompts[w].size() <= common) {
                fprintf(stderr, "%s: skip duplicate %s\n", __func__, words[w].c_str());
                continue;
            }
            std::vector<std::pair<std::string, double>> out;
            if (!run_word(prompts[w], out)) {
                return 1;
            }
            // 每个词单独打印：第一行是词，后面是它的候选
            printf("%s\n", words[w].c_str());
            for (const auto & c : out) {
                printf("%s\t%.2f\n", c.first.c_str(), c.second);
            }
            printf("\n");
        }
    }

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n", __func__, n_decode,
            (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_batch_ext_free(batch);
    llama_sampler_free(subset);
    llama_sampler_free(smpl);
    llama_sampler_free(grmr);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}

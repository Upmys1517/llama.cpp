#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <map>

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
static int32_t batch_add_one(llama_batch_ext * batch, llama_seq_id seq, llama_token tok, llama_pos pos) {
    const int32_t idx = llama_batch_ext_add_token(batch, seq, tok);
    if (idx < 0) {
        return idx;
    }
    if (!llama_batch_ext_set_pos(batch, idx, &pos)) {
        return -1;
    }
    if (!llama_batch_ext_set_output_logits(batch, idx, true)) {
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
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
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
    const chat_message system_message = { "system", sys };
    std::vector<chat_message> history = {
        system_message, { "user", prompt }
    };
    prompt = format_chat(model, history);

    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) <
        0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_seq_max = 2 * n_beam + 1;
    ctx_params.kv_unified = true;
    ctx_params.n_ctx     = n_prompt + 2 * n_beam * n_predict;  
    ctx_params.n_batch   = std::max(n_prompt, n_beam);
    ctx_params.no_perf   = false;

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

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int  n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch_ext * batch    = llama_batch_ext_init(ctx);
    

    // main loop

    const auto t_main_start = ggml_time_us();
    int        n_decode     = 0;

    // 提示词只算一次，放进模板序列 0
    std::vector<int> free_ids; 
    for (int i = 1; i <= 2 * n_beam; ++i) {
        free_ids.push_back(i);
    }

    llama_batch_ext_clear(batch);
    batch_set_tokens(batch, prompt_tokens.data(), n_prompt, 0);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
        fprintf(stderr, "%s: error: llama_process failed\n", __func__);
        return 1;
    }
    const float * root_logits = llama_get_logits_ith(ctx, -1);

    // 虚拟的起点：一条内容为空、就绪位置是提示词末尾的路，序列号是 0 （模板）
    std::vector<beam> live;
    live.push_back({"", 0.0f, llama_sampler_clone(grmr), 0, n_prompt, root_logits});
    std::map<std::string, double> finished;     // 文字 -> 最高分

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

            // 用这条路自己的语法状态筛掉违规 token
            std::vector<llama_token_data> cur(tag_vocab.size());
            for (int i = 0; i < static_cast<int>(tag_vocab.size()); ++i) {
                const llama_token id = tag_vocab[i];
                cur[i] = { id, logits[id], 0.0f };
            }
            llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
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
            std::partial_sort(ok.begin(), ok.begin() + n_ok, ok.end(), [](const auto & x, const auto & y) {
                return x.first > y.first; 
            });
            for (int j = 0; j < n_ok; ++j) {
                pool.push_back({live[b].score + ok[j].first, b, ok[j].second });
            }
        }

        std::sort(pool.begin(), pool.end(), [](const cand & x, const cand & y) {
            return x.score > y.score; 
        });
        
        // 只给活下来的 K 个建新序列
        std::vector<beam> next;
        std::vector<int32_t> batch_ids;

        llama_memory_t mem = llama_get_memory(ctx);
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
            // 给子路分一个空闲序列，把父路的全部记忆（提示词 + 父路已写的 token）挂过来
            const llama_seq_id id = free_ids.back();
            free_ids.pop_back();
            llama_memory_seq_rm(mem, id, -1, -1);
            llama_memory_seq_cp(mem, p.seq, id, -1, -1);

            // 只把这一步新选的 token 放进 batch，位置接在父路后面
            const int32_t idx = batch_add_one(batch, id, pool[c].tok, p.next_pos);
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
            nb.logits = nullptr; // 下面算完再填
            next.push_back(nb);
        }

        // 所有子路的新 token 放在同一个 batch 里，一次算完
        if (!batch_ids.empty()) {
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                fprintf(stderr, "%s: error: llama_process failed\n", __func__);
                return 1;
            }
            n_decode += batch_ids.size();
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

    std::vector<std::pair<std::string, double>> out(finished.begin(), finished.end());
    std::sort(out.begin(), out.end(), [](const auto & x, const auto & y) { return x.second > y.second; });
    for (int i = 0; i < static_cast<int>(out.size()) && i < n_beam; ++i) {
        printf("%s\t%.2f\n", out[i].first.c_str(), out[i].second);
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n", __func__, n_decode,
            (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_sampler_free(grmr);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}

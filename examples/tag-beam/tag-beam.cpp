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
    std::vector<llama_token> toks;  // 这条路已经写了哪些 token
    std::string              text;  // 对应的文字
    double                   score; // 每步 log 概率之和
    llama_sampler *          grmr;  // 每条路自己的语法状态
};

// 笨方法：清空记忆，把「提示词 + 这条路的 token」从头算一遍，返回最后一个位置的 logits
static const float * eval_from_scratch(llama_context * ctx, llama_batch_ext * batch,
                                       const std::vector<llama_token> & prompt,
                                       const std::vector<llama_token> & toks) {
    // 把 prompt 和 toks 拼成一个 vector all
    std::vector<llama_token> all;
    all.reserve(prompt.size() + toks.size());
    all.insert(all.end(), prompt.begin(), prompt.end());
    all.insert(all.end(), toks.begin(), toks.end());

    // 清空记忆
    llama_memory_clear(llama_get_memory(ctx), true);

    // batch_set_tokens(batch, all.data(), all.size(), 0)，再 llama_process；失败就 return nullptr
    batch_set_tokens(batch, all.data(), all.size(), 0);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
        return nullptr;
    }
    return llama_get_logits_ith(ctx, -1);
}

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
    // n_ctx is the context size
    ctx_params.n_ctx                = n_prompt + n_predict;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch              = n_prompt + n_predict;
    // enable performance counters
    ctx_params.no_perf              = false;

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

    const int K = 4;
    std::vector<beam> live;
    live.push_back({ {}, "", 0.0, llama_sampler_clone(grmr) });
    std::map<std::string, double> finished;     // 文字 -> 最高分

    struct cand {
        double      score;   // 走到这一步的总分
        int         parent;  // 从 live 里第几条路延申出来
        llama_token tok;     // 这一步选的 token
    };

    while (!live.empty()) {
        std::vector<cand> pool;
        for (int b = 0; b < static_cast<int>(live.size()); ++b) {
            const float * logits = eval_from_scratch(ctx, batch, prompt_tokens, live[b].toks);
            if (logits == nullptr) {
                fprintf(stderr, "%s: eval failed\n", __func__);
                return 1;
            }
            n_decode += 1;
            const int n_vocab = llama_vocab_n_tokens(vocab);

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
            std::vector<double> lp(n_vocab);
            for (int i = 0; i < n_vocab; ++i) {
                lp[i] = logits[i] - max_l - log_sum;
            }

            // 用这条路自己的语法状态筛掉违规 token
            std::vector<llama_token_data> cur(n_vocab);
            for (int i = 0; i < n_vocab; ++i) {
                cur[i] = { i, logits[i], 0.0f };
            }
            llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(live[b].grmr, &arr);

            // 允许的里面，按 lp 取前 K 个
            std::vector<std::pair<double, llama_token>> ok;
            for (int j = 0; j < static_cast<int>(arr.size); ++j) {
                if (arr.data[j].logit != -INFINITY) {
                    ok.push_back({ lp[arr.data[j].id], arr.data[j].id});
                }
            }
            const int n_ok = std::min<int>(K, ok.size());
            std::partial_sort(ok.begin(), ok.begin() + n_ok, ok.end(),
                              [](const auto & x, const auto & y) {
                                return x.first > y.first;
                              });
            for (int j = 0; j < n_ok; ++j) {
                pool.push_back({ live[b].score + ok[j].first, b, ok[j].second });
            }
        }

        // 所有路的候选合在一起，按总分排序
        std::sort(pool.begin(), pool.end(), 
                  [](const cand & x, const cand & y) { 
                    return x.score > y.score; 
                });

        std::vector<beam> next;
        for (int c = 0; c < std::min<int>(K, pool.size()); ++c) {
            const beam & p = live[pool[c].parent];
            if (llama_vocab_is_eog(vocab, pool[c].tok)) {
                auto it = finished.find(p.text);
                if (it == finished.end() || it->second < pool[c].score) {
                    finished[p.text] = pool[c].score;
                }
            } else {
                beam nb;
                nb.toks = p.toks;
                nb.toks.push_back(pool[c].tok);
                char buf[128];
                const int n = llama_token_to_piece(vocab, pool[c].tok, buf, sizeof(buf), 0, true);
                nb.text = p.text + std::string(buf, n);
                nb.score = pool[c].score;
                nb.grmr = llama_sampler_clone(p.grmr);
                llama_sampler_accept(nb.grmr, pool[c].tok);
                next.push_back(std::move(nb));
            }
        }

        for (auto & b : live) {
            llama_sampler_free(b.grmr);
        }
        live = std::move(next);
    }

    // 打印前 K 个已完成的
    std::vector<std::pair<std::string, double>> out(finished.begin(), finished.end());
    std::sort(out.begin(), out.end(), [](const auto & x, const auto & y) {return x.second > y.second; });
    for (int i = 0; i < static_cast<int>(out.size()) && i < static_cast<int>(K); ++i) {
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

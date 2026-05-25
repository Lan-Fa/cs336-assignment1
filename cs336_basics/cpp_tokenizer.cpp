//
// Created by r on 26-5-25.
//

#include "cpp_tokenizer.hpp"

#include <queue>

namespace BPE {
    PairId::PairId() : first(-1), second(-1) {
    }

    PairId::PairId(int first, int second) : first(first), second(second) {
    }

    bool PairId::operator==(const PairId &o) const {
        return first == o.first && second == o.second;
    }

    std::size_t PairIdHash::operator()(const PairId &p) const {
        const auto h1 = std::hash<int>{}(p.first);
        const auto h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }

    CppBPE::CppBPE(pybind11::dict py_token_to_id, pybind11::list merges) {
        for (auto item: py_token_to_id) {
            const std::string token =
                    pybind11::reinterpret_borrow<pybind11::bytes>(item.first);

            const int id = pybind11::cast<int>(item.second);

            this->token_to_id[token] = id;
        }

        for (std::size_t rank = 0; rank < merges.size(); ++rank) {
            pybind11::tuple pair = pybind11::cast<pybind11::tuple>(merges[rank]);

            if (pair.size() != 2) {
                throw std::runtime_error("merge pair must have length 2");
            }

            const std::string left =
                    pybind11::reinterpret_borrow<pybind11::bytes>(pair[0]);

            const std::string right =
                    pybind11::reinterpret_borrow<pybind11::bytes>(pair[1]);

            const int left_id = id_of_token(left);
            const int right_id = id_of_token(right);

            const std::string merged = left + right;
            const int merged_id = id_of_token(merged);

            PairId pid(left_id, right_id);

            this->pair_rank[pid] = static_cast<int>(rank);
            this->pair_merge_id[pid] = merged_id;
        }
    }

    int CppBPE::id_of_token(const std::string &token) const {
        auto it = this->token_to_id.find(token);

        if (it == this->token_to_id.end()) {
            throw std::runtime_error("token not found in vocab");
        }

        return it->second;
    }

    std::vector<int> CppBPE::encode_bytes(pybind11::bytes input_bytes) const {
        const std::string input = input_bytes;

        auto tokens = split_bytes(input);
        tokens = merge_tokens(tokens);

        return tokens;
    }

    std::vector<int> CppBPE::split_bytes(const std::string &input) const {
        std::vector<int> res;
        res.reserve(input.size());

        for (byte c: input) {
            const std::string token(1, static_cast<char>(c));
            res.push_back(id_of_token(token));
        }

        return res;
    }

    std::vector<int> CppBPE::merge_tokens(const std::vector<int> &tokens) const {
        const int n = static_cast<int>(tokens.size());

        if (n <= 1) {
            return tokens;
        }

        std::vector<int> val(n);
        std::vector<int> nxt(n, -1), pre(n, -1);
        std::vector<bool> alive(n, true);

        int head = 0;
        for (int i = 0; i < n; i++) {
            val[i] = tokens[i];
            if (i > 0) {
                pre[i] = i - 1;
            }
            if (i < n - 1) {
                nxt[i] = i + 1;
            }
        }

        struct obj {
            int p, rk;
            int left_id, right_id, merged_id;
        };

        auto cmp = [&](const obj &l, const obj &r) -> bool {
            if (l.rk != r.rk) return l.rk > r.rk;
            return l.p > r.p;
        };

        std::priority_queue<obj, std::vector<obj>, decltype(cmp)> pq(cmp);

        auto push_pair = [&](int p) -> void {
            if (p == -1 || !alive[p]) {
                return;
            }
            int q = nxt[p];
            if (q == -1 || !alive[q]) {
                return;
            }
            PairId pid(val[p], val[q]);
            auto it_rank = this->pair_rank.find(pid);
            if (it_rank == this->pair_rank.end()) {
                return;
            }
            auto it_merge = this->pair_merge_id.find(pid);
            if (it_merge == this->pair_merge_id.end()) {
                throw std::runtime_error("merge id not found");
            }

            pq.push({
                p,
                it_rank->second,
                val[p],
                val[q],
                it_merge->second
            });
        };

        for (int i = 0; i < n - 1; i++) {
            push_pair(i);
        }

        while (!pq.empty()) {
            obj cur = pq.top();
            pq.pop();

            int p = cur.p;

            if (p == -1 || !alive[p]) {
                continue;
            }
            int q = nxt[p];
            if (q == -1 || !alive[q]) {
                continue;
            }
            if (val[p] != cur.left_id || val[q] != cur.right_id) {
                continue;
            }

            PairId pid(val[p], val[q]);
            auto it_rank = this->pair_rank.find(pid);
            if (it_rank == this->pair_rank.end()) {
                continue;
            }
            if (it_rank->second != cur.rk) {
                continue;
            }

            val[p] = cur.merged_id;
            alive[q] = false;
            int nq = nxt[q];
            nxt[p] = nq;
            if (nq != -1) {
                pre[nq] = p;
            }
            push_pair(pre[p]);
            push_pair(p);
        }

        std::vector<int> res;
        for (int p = head; p != -1; p = nxt[p]) {
            if (alive[p]) {
                res.push_back(val[p]);
            }
        }
        return res;
    }
}

PYBIND11_MODULE(_cpp_tokenizer, m) {
    pybind11::class_<BPE::CppBPE>(m, "CppBPE")
        .def(pybind11::init<pybind11::dict, pybind11::list>())
        .def("encode_bytes", &BPE::CppBPE::encode_bytes);
}
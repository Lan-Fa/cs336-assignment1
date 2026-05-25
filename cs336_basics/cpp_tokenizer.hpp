#ifndef CPP_TOKENIZER_HPP
#define CPP_TOKENIZER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cstddef>

namespace BPE {
    using byte = unsigned char;

    struct PairId {
        int first;
        int second;

        PairId();
        PairId(int first, int second);

        bool operator==(const PairId &o) const;
    };

    struct PairIdHash {
        std::size_t operator()(const PairId &p) const;
    };

    class CppBPE {
        std::unordered_map<std::string, int> token_to_id;

        // (left_id, right_id) -> rank
        std::unordered_map<PairId, int, PairIdHash> pair_rank;

        // (left_id, right_id) -> merged_id
        std::unordered_map<PairId, int, PairIdHash> pair_merge_id;

    public:
        CppBPE(pybind11::dict token_to_id, pybind11::list merges);

        [[nodiscard]] std::vector<int> encode_bytes(pybind11::bytes input_bytes) const;

    private:
        [[nodiscard]] int id_of_token(const std::string &token) const;

        [[nodiscard]] std::vector<int> split_bytes(const std::string &input) const;

        [[nodiscard]] std::vector<int> merge_tokens(
            const std::vector<int> &tokens
        ) const;
    };
}

#endif // CPP_TOKENIZER_HPP
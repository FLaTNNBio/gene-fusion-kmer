#include <vector>
#include <string>
#include <map>

using namespace std;

#ifndef GENE_FUSION_TOKENZER_H
#define GENE_FUSION_TOKENZER_H

map <string, string> specialTokens = {
        {"unk_token",  "[UNK]"},
        {"sep_token",  "[SEP]"},
        {"pad_token",  "[PAD]"},
        {"cls_token",  "[CLS]"},
        {"mask_token", "[MASK]"},
};

class Vocab {
public:
    int lenKmer;
    string vocabPathFile;
    map<string, int> vocab;

    Vocab(int kmer);
    void encodeSentences(vector<string> sentences, int maxLen);

private:
    void createVocab(int lenKmer, string vocabPathFile);
    map<string, int> loadVocab();

};

#endif //GENE_FUSION_TOKENZER_H

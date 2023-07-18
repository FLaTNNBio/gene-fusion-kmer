#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#include "tokenizer.h"

Vocab::Vocab(int kmer) {
    lenKmer = kmer;
    ostringstream vocabFileName;
    vocabFileName << "vocab_" << lenKmer << ".txt";
    vocabPathFile = vocabFileName.str();

    createVocab(lenKmer, vocabPathFile);
    vocab = loadVocab();
}

void printAllKLengthRec(ostream &outputFile, char set[], string prefix, int n, int k) {
    // base case: k is 0, print prefix
    if (k == 0) {
        outputFile << (prefix) << endl;
        return;
    }

    // one by one add all characters from set and recursively call for k equals to k-1
    for (int i = 0; i < n; i++) {
        string newPrefix;
        // next character of input added
        newPrefix = prefix + set[i];
        // k is decreased, because we have added a new character
        printAllKLengthRec(outputFile, set, newPrefix, n, k - 1);
    }

}

void Vocab::createVocab(int lenKmer, string vocabPathFile) {
    // init the alphabet
    char alphabet[] = {
            'A', 'C', 'G', 'T', 'N'
    };

    // check if file exists
    ifstream inputFile(vocabPathFile);
    if (inputFile.good()) {
        return;
    }
    inputFile.close();

    // open vocab file
    ofstream outputFile;
    outputFile.open(vocabPathFile);

    // create vocab
    for (const auto &[key, value]: specialTokens) {
        outputFile << value << endl;
    }
    printAllKLengthRec(outputFile, alphabet, "", 5, lenKmer);

    // close file
    outputFile.close();
}

map<string, int> Vocab::loadVocab() {
    map<string, int> vocab;
    // read file line by line
    ifstream inputFile(vocabPathFile);
    string line;
    for (int i = 0; getline(inputFile, line); i++) {
        if (line.compare("") != 0) {
            vocab[line] = i;
        }
    }
    inputFile.close();

    return vocab;
}

void Vocab::encodeSentences(vector<string> sentences, int maxLen) {
    string delimiter = " ";
    // for each sentence in sentences vector
    for (string sentence : sentences) {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        string token;
        vector<string> words;
        // split sentence in words vector
        while ((pos_end = sentence.find(delimiter, pos_start)) != string::npos) {
            token = sentence.substr(pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            transform(token.begin(), token.end(), token.begin(), ::toupper);
            words.push_back(token);
        }
        words.push_back(sentence.substr(pos_start));

        // for each word of sentences
        vector<int> input_ids;
        input_ids.push_back(vocab[specialTokens["cls_token"]]);
        for (string word : words) {
            // if vocab contain this word
            if (vocab.count(word)) {
                input_ids.push_back(vocab[word]);
            } else {
                input_ids.push_back(vocab[specialTokens["unk_token"]]);
            }
        }
        for (int i = maxLen - encodedSentence.size() - 1; i > 0; i--) {
            input_ids.push_back(vocab[specialTokens["pad_token"]]);
        }
        input_ids.push_back(vocab[specialTokens["sep_token"]]);

        // init attention_mask vector
        vector<int> attention_mask;


        for (int encoded : input_ids) {
            cout << encoded << " ";
        }
        cout << endl;
    }

}

int main() {
    Vocab vocab = Vocab(6);
    vector<string> sentences = {
            "AAAAAA AGGACC AAATCA AANAAA AAAAAA AAGAAA AAAACT"
    };
    vocab.encodeSentences(sentences, 20);
    return 0;
}
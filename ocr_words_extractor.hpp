#pragma once

#include <assert.h> 
#include <string_view>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

#include <leptonica/allheaders.h>

#include "ocr_classifier.hpp"
#include "json.hpp"

namespace fs = std::filesystem;

class Ocr_words_extractor {
public :
	Ocr_words_extractor(std::string_view dir) : dir_name_(dir) {
		assert(fs::is_directory(dir));
	}

	void process(std::string_view lang = "eng", int num_threads = 1, float aquisition_rate = 0.2f) {
		
		Ocr_classifier ocr_classifier(lang, num_threads);
		std::vector<std::map<std::string, int>> words_maps;
		
		for (const auto& entry : fs::directory_iterator(dir_name_)) {
			if (fs::is_directory(entry)) {
				classes_.emplace_back(entry.path().filename().string());
				
				//iterating over same class documents
				std::vector<std::vector<std::string>> documents_words;
				int treated_documents = 0;
				for (const auto& sub_entry : fs::directory_iterator(entry)) {
					if (is_supported_image_file_extension(sub_entry.path().extension().string())) {
						Pix* image = pixRead(sub_entry.path().string().c_str());
						auto result_str = ocr_classifier.apply_ocr(image);
						documents_words.emplace_back(split_document_words(result_str));
						pixDestroy(&image);
						treated_documents++;
					}
				}
				auto words_count_map = merge_document_words(documents_words);
				
				std::vector<std::string> tmp;
				auto min_repeats = treated_documents * aquisition_rate; //maybe make it a parameter
				for (auto& el : words_count_map) {
					if (el.second >= min_repeats) {
						tmp.emplace_back(el.first);
					}
				}
				class_words_.emplace_back(std::move(tmp));
			}
		}
	}

	std::map<std::string, int> merge_document_words(std::vector<std::vector<std::string>> documents_words) {
		std::map<std::string, int> res_map;
		for (auto& doc_words : documents_words) {
			for (auto& word : doc_words) {
				if (res_map.find(word) == res_map.end()) {
					res_map[word] = 0;
				}
				else {
					res_map[word]++;
				}
			}
		}
		return res_map;
	}

	std::vector<std::string> split_document_words(std::string ocr_result) {
		std::vector<std::string> result;
		std::vector<std::string> lines = split_string(ocr_result, '\n');
		for (auto& line : lines) {

			std::vector<std::string> captured_words;
			captured_words = split_string(line, ' ');
			if (captured_words.size() == 1) {
				result.emplace_back(captured_words[0]);
			}
			else if (captured_words.size() != 0) {
				for (int i = 0; i < captured_words.size() - 1; i++) {
					result.emplace_back(captured_words[i] + ' ' + captured_words[i + 1]);
				}
			}
			
		}
		std::sort(result.begin(), result.end());
		result.erase(std::unique(result.begin(), result.end()), result.end());
		return result;
	}

	void save(std::string_view path) {
		nlohmann::json j;
		j["classes"] = classes_;
		j["words"] = class_words_;
		std::ofstream ofs(path);
		ofs << j.dump();
		ofs.close();
	}

private :
	std::string_view dir_name_;
	std::vector<std::string> classes_;
	std::vector<std::vector<std::string>> class_words_;
};
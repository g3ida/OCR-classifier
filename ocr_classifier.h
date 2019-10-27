#pragma once
#include <leptonica/allheaders.h>
#include <vector>
#include <string>
#include <map>
#include <tesseract/baseapi.h>
#include "utils.hpp"
#include <cstdio>
#include <execution>
#include <mutex>
#include <chrono>
#include "east_detector.h"
#include <future>


class Ocr_classifier {
public :

	Ocr_classifier(std::string_view lang, int num_threads = 1) : lang_(lang) {
		
		api_ = new tesseract::TessBaseAPI();
		if (api_->Init(NULL, lang.data())) {
			fprintf(stderr, "Could not initialize tesseract.\n");
			throw "could not inititialize tesseract";
		}

		if (num_threads > 1) {
			for (int i = 0; i < num_threads-1; i++) {
				auto api = new tesseract::TessBaseAPI();
				if (api->Init(NULL, lang.data())) {
					fprintf(stderr, "Could not initialize tesseract.\n");
					throw "could not inititialize tesseract";
				}
				sub_api_.emplace_back(std::move(api));
			}
		}
	}
	
	void set_classes(std::map<std::string, std::vector<std::string>> classes) {
		for (auto [cls, words] : classes) {
			classes_.push_back(cls);
			class_words_.push_back(words);
		}
	}

	std::string classifiy(PIX* image) {

		std::string text;
		if (sub_api_.size() > 0) {
			text = apply_line_ocr_(image);
		}
		else {
			text = apply_full_ocr_(image);
		}

		for (int i = 0; i < classes_.size(); i++) {
			int ocrs = 0;
			for (auto& word : class_words_[i]) {
				if (text.find(word) != -1) {
					ocrs++;
					std::cout << "found " << word << std::endl;
				}
			}
			occurences_.push_back(ocrs);
		}
		for (auto i : occurences_) {
			std::cout << "=" << i << std::endl;
		}
		return classes_[std::distance(occurences_.begin(), std::max_element(occurences_.begin(), occurences_.end()))];
	}


	~Ocr_classifier() {
		delete api_;
	}
private :

	std::string apply_full_ocr_(PIX* image) {
		auto start = std::chrono::high_resolution_clock::now();
		
		api_->SetImage(image);
		char* outText = api_->GetUTF8Text();
		std::string s(outText);
		delete[] outText;

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "ocr time : "<< duration.count() << std::endl;
		
		return s;
	}

	std::string apply_line_ocr_(PIX* image) {
		auto start = std::chrono::high_resolution_clock::now();
		
		auto max_boxes_h = image->h / 300ULL;
		auto boxes_h = std::min(max_boxes_h, image->h / (sub_api_.size() + 1));

		auto step = image->h / boxes_h;
		auto padding = 75;
		BOX* region = boxCreate(0, 0, image->w, step + padding);
		PIX* imgCrop = pixClipRectangle(image, region, NULL);
		api_->SetImage(imgCrop);
		
		std::vector<std::future<std::string>> futures;

		futures.emplace_back(std::async(std::launch::async,
			[&]() {
			char* text = api_->GetUTF8Text();
			std::string result(text);
			delete[] text;
			return result;
		}));
		
		for (int i = 0; i < sub_api_.size(); i++) {
			BOX* region = boxCreate(0, (i + 1) * step, image->w, (i + 2) * step + padding);
			auto api = sub_api_[i];
			futures.emplace_back(std::async(std::launch::async,
				[=]() {
				PIX* imgCrop = pixClipRectangle(image, region, NULL);
				api->SetImage(imgCrop);
				char* text = api->GetUTF8Text();
				std::string result(text);
				delete[] text;
				return result;
			}));
		}

		std::string result_string;
		for (auto& f : futures) {
			auto str = f.get();
			std::cout << str << std::endl;
			result_string.append(str);
		}
		
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "ocr time : " << duration.count() << std::endl;
		
		return result_string;
	}

	std::vector<tesseract::TessBaseAPI*> sub_api_;
	std::string lang_ = "eng";
	std::vector<std::string> classes_;
	std::vector<std::vector<std::string>> class_words_;
	std::vector<int> occurences_;
	tesseract::TessBaseAPI* api_ = nullptr;
};
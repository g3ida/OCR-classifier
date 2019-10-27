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


class Ocr_classifier {
public :
	Ocr_classifier(std::string_view lang) : lang_(lang) {
		api_ = new tesseract::TessBaseAPI();
		if (api_->Init(NULL, lang.data())) {
			fprintf(stderr, "Could not initialize tesseract.\n");
			exit(1);
		}
	}
	
	void set_classes(std::map<std::string, std::vector<std::string>> classes) {
		for (auto [cls, words] : classes) {
			classes_.push_back(cls);
			class_words_.push_back(words);
		}
	}

	std::string classifiy(PIX* image) {
		auto text = apply_full_ocr_(image);
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
		
		api_->SetImage(image);
		Boxa* boxes = api_->GetComponentImages(tesseract::RIL_TEXTLINE, true, NULL, NULL);
		int epsilon = 5;
		if (boxes) {
			printf("Found %d textline image components.\n", boxes->n);
			std::vector<std::string> results;
			std::mutex m;
			std::for_each(
				std::execution::par_unseq,
				boxes->box,
				boxes->box + boxes->n,
				[&](auto&& b)
			{
				tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
				if (api->Init(NULL, lang_.data())) {
					auto box = boxClone(b);
					api->SetImage(pixClipRectangle(image, box, NULL));
					char* ocrResult = api_->GetUTF8Text();
					int conf = api_->MeanTextConf();
					std::string s(ocrResult);
					m.lock();
					results.push_back(ocrResult);
					m.unlock();
					fprintf(stdout, "Box[]: x=%d, y=%d, w=%d, h=%d, confidence: %d, text: %s",
						box->x, box->y, box->w, box->h, conf, ocrResult);
					delete[] ocrResult;
					delete api;
				}
			});
			
			std::string returned;
			for (const auto& elem : results) returned += '\n' + elem;
		
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			std::cout << "ocr time : " << duration.count() << std::endl;
			
			return returned;
		
		
		}
	}


	std::string lang_ = "eng";
	std::vector<std::string> classes_;
	std::vector<std::vector<std::string>> class_words_;
	std::vector<int> occurences_;
	tesseract::TessBaseAPI* api_ = nullptr;
};
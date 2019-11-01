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
#include <future>
#include "east_detector.h"

class Ocr_classifier {
public :

	Ocr_classifier(std::string_view lang, int num_threads = 1, bool use_east_detector = false) : 
		lang_(lang), use_east_detector_(use_east_detector) {
		
		for (int i = 0; i < num_threads; i++) {
			auto api = new tesseract::TessBaseAPI();
			if (api->Init(NULL, lang.data())) {
				fprintf(stderr, "Could not initialize tesseract.\n");
				throw "could not inititialize tesseract";
			}
			sub_api_.emplace_back(std::move(api));
		}
	}
	
	void set_classes(std::map<std::string, std::vector<std::string>> classes) {
		for (auto [cls, words] : classes) {
			classes_.push_back(cls);
			class_words_.push_back(words);
		}
	}

	std::string classifiy(PIX* image) {

		std::string text = use_east_detector_ ? apply_ocr_after_east_detector(image) : apply_ocr(image);

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
		for (auto api : sub_api_) {
			api->End();
			delete api;	
		}
	}

	std::string apply_ocr(PIX* image) {
		auto start = std::chrono::high_resolution_clock::now();

		auto max_boxes_h = image->h / 300ULL;
		auto boxes_h = std::min(max_boxes_h, sub_api_.size());

		auto step =  boxes_h == 0 ? image->h : image->h / boxes_h;
		auto padding = 75;

		std::vector<std::future<std::string>> futures;
		
		for (int i = 0; i < boxes_h; i++) {
			BOX* region = boxCreate(0, (i) * step, image->w, (i + 1) * step + padding);
			auto api = sub_api_[i];
			futures.emplace_back(std::async(std::launch::async,
				[=]() {
				PIX* imgCrop = pixClipRectangle(image, region, NULL);				
				api->SetImage(imgCrop);
				char* text = api->GetUTF8Text();
				std::string result(text);
				pixDestroy(&imgCrop);
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



	std::string apply_ocr_after_east_detector(PIX* image) {
		// initialization should be splitted from here
		EAST_detector detector(320);
		detector.load_model("frozen_east_text_detection.pb");
		auto mat = pix1_to_mat(image);
		cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR); //just a hack need to get back color
		auto detection = detector.detect(mat);
		std::string result;
		for (auto rec : detection) {
			auto roi = mat(rect_add_margin(rec.boundingRect(), 5));
			cv::cvtColor(roi, roi, cv::COLOR_BGR2RGBA);
			sub_api_[0]->SetImage(roi.data, roi.cols, roi.rows, 4, 4 * roi.cols);
			char* text = sub_api_[0]->GetUTF8Text();
			result.append(text);
			delete[] text;
		}
		return result;
	}

private:

	std::vector<tesseract::TessBaseAPI*> sub_api_;
	std::string lang_ = "eng";
	std::vector<std::string> classes_;
	std::vector<std::vector<std::string>> class_words_;
	std::vector<int> occurences_;
	bool use_east_detector_ = false;
};
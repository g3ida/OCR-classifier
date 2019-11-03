#pragma once
#include <leptonica/allheaders.h>
#include <vector>
#include <string>
#include <map>
#include <tesseract/baseapi.h>
#include "utils.hpp"
#include <cstdio>
#include <execution>
#include <algorithm>
#include <mutex>
#include <chrono>
#include <future>
#include <mutex>
#include <shared_mutex>
#include "east_detector.h"

class Ocr_classifier {
public :

	Ocr_classifier(std::string_view lang, int num_threads = 1, bool use_early_stopping = false) :
		lang_(lang), use_early_stopping_(use_early_stopping) {
		
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

		std::string text = use_early_stopping_ ? ocr_with_early_stopping(image) : apply_ocr(image);
		reset_occurences_list();
		match_occurences(text);
		for (auto i = 0; i < occurences_.size(); i++) {
			std::cout << classes_[i] << " score : " << occurences_[i] << '/' << class_words_[i].size() 
				<< " (" << (float)occurences_[i]/ class_words_[i].size()  << ")" <<std::endl;
		}
		return classes_[std::distance(occurences_.begin(), std::max_element(occurences_.begin(), occurences_.end()))];
	}

	void reset_occurences_list() {
		total_matched_ = 0;
		occurences_.clear();
		occurences_.resize(class_words_.size(), 0);
		found_words_.clear();
		found_words_.resize(class_words_.size(), std::vector<bool>());
		for (int i = 0; i < class_words_.size(); i++) {
			found_words_[i].resize(class_words_[i].size(), false);
		}
	}

	void match_occurences(std::string text) {
		for (int i = 0; i < classes_.size(); i++) {
			int ocrs = 0;
			for (int j = 0; j < class_words_[i].size(); j++) {
				if (text.find(class_words_[i][j]) != -1) {
					ReadLock rl(words_loc);
					if (found_words_[i][j] == false) {
						rl.unlock();
						ocrs++;
						WriteLock wl(words_loc);
						found_words_[i][j] = true;
					}
					//std::cout << "matched : " << word << std::endl;
				}
			}
			WriteLock write_lock{ occurences_loc };
			occurences_[i] += ocrs;
			total_matched_ += ocrs;
			write_lock.unlock();

			//setup the max and the second matched class needed for early stopping estimation.
			ReadLock read_lock{ occurences_loc };

			if (occurences_[i] > occurences_[most_matched_class_]) {
				read_lock.unlock();
				write_lock.lock();
				second_most_matched_class_ = most_matched_class_;
				most_matched_class_ = i;
			}
			else if (occurences_[i] > occurences_[second_most_matched_class_]) {
				read_lock.unlock();
				if (i != most_matched_class_) {
					write_lock.lock();
					second_most_matched_class_ = i;
				}
			}
		}
	}


	~Ocr_classifier() {
		for (auto api : sub_api_) {
			api->End();
			delete api;	
		}
	}

	void ocr_on_boxa(tesseract::TessBaseAPI* api, Boxa* boxes) {
		for (int i = 0; i < boxes->n; i++) {
			BOX* box = boxaGetBox(boxes, i, L_CLONE);
			api->SetRectangle(box->x, box->y, box->w, box->h);
			char* ocrResult = api->GetUTF8Text();
			match_occurences(ocrResult);
			delete[] ocrResult;
			ReadLock read_lock(occurences_loc);

			if (early_stopping_verified()) {
				break;
			}
		}
	}


	inline bool early_stopping_verified() {
		return  (occurences_[most_matched_class_] > class_words_[most_matched_class_].size()*0.1f &&
			occurences_[most_matched_class_] - occurences_[second_most_matched_class_] > occurences_[second_most_matched_class_]);
	}

	std::string ocr_with_early_stopping(PIX* input_image) {
		auto start = std::chrono::high_resolution_clock::now();
		reset_occurences_list();

		//need to split image here
		auto boxes_h = input_image->h / 512ULL;
		auto step = boxes_h == 0 ? input_image->h : input_image->h / boxes_h;
		auto padding = 32;
		for (int k = 0; k < boxes_h; k++) {
			BOX* region = boxCreate(0, (k)* step, input_image->w, (k + 1) * step + padding);
			PIX* image = pixClipRectangle(input_image, region, NULL);
			sub_api_[0]->SetImage(image);
			Boxa* boxes = sub_api_[0]->GetComponentImages(tesseract::RIL_TEXTLINE, true, NULL, NULL);
			 
			if (boxes == nullptr) {
				pixDestroy(&image);
				continue;
			}
			auto boxes_per_api = boxes->n / sub_api_.size();

			std::vector<std::future<void>> api_futures_;
			int i = 1;
			if (boxes_per_api != 0) {
				for (; i < sub_api_.size(); i++) {
					api_futures_.emplace_back(std::async(std::launch::async, [=]() {
						Boxa b;
						b.box = boxes->box + (i - 1) * boxes_per_api;
						b.n = boxes_per_api;
						b.nalloc = boxes->nalloc;
						b.refcount = boxes->refcount;
						sub_api_[i]->SetImage(image);
						ocr_on_boxa(sub_api_[i], &b);
					}));
				}
			}

			Boxa b;
			b.box = boxes->box + (i - 1) * boxes_per_api;
			b.n = boxes->n - boxes_per_api * (sub_api_.size() - 1);
			b.nalloc = boxes->nalloc;
			b.refcount = boxes->refcount;
			ocr_on_boxa(sub_api_[0], &b);

			for (auto& future : api_futures_) {
				future.get();
			}
			pixDestroy(&image);
			//check for early stopping
			if (early_stopping_verified()) {
				break;
			}
		}

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "ocr time : " << duration.count() << std::endl;

		//show results;
		for (auto i = 0; i < occurences_.size(); i++) {
			std::cout << classes_[i] << " score : " << occurences_[i] << '/' << class_words_[i].size()
				<< " (" << (float)occurences_[i] / class_words_[i].size() << ")" << std::endl;
		}
		return classes_[std::distance(occurences_.begin(), std::max_element(occurences_.begin(), occurences_.end()))];

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
			//std::cout << str << std::endl;
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
	std::vector<std::vector<bool>> found_words_;
	int total_matched_ = 0;
	int most_matched_class_ = 0;
	int second_most_matched_class_ = 0;
	std::vector<int> occurences_;
	bool use_east_detector_ = false;
	bool use_early_stopping_ = false;

	typedef std::shared_mutex Lock;
	typedef std::unique_lock<Lock>  WriteLock;
	typedef std::shared_lock<Lock>  ReadLock;
	Lock occurences_loc;
	Lock words_loc;
};
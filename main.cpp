#include "CLI11.hpp"
#include "json.hpp"
#include <leptonica/allheaders.h>
#include <iostream>

#include "ocr_classifier.h"


int main(int argc, char* argv[]) {
	CLI::App app{ "OCR Classifier" };

	std::string config_filename;
	std::string image_filename;
	int workers = 4;

	app.add_option("-c,--config", config_filename, "configuration file")->check(CLI::ExistingFile)->required();
	app.add_option("-i,--image", image_filename, "image file")->check(CLI::ExistingFile)->required();
	app.add_option("-w,--workers", workers, "workers threads")->check(CLI::PositiveNumber);

	CLI11_PARSE(app, argc, argv);

	std::ifstream config(config_filename);
	nlohmann::json j;
	config >> j;


	std::vector<std::string> classes;
	std::vector<std::vector<std::string>>  words;
	j.at("classes").get_to(classes);
	j.at("words").get_to(words);

	std::map<std::string, std::vector<std::string>> map;
	// create map
	std::transform(classes.begin(), classes.end(), words.begin(), std::inserter(map, map.end()), 
		[](auto a, auto b) { return std::make_pair(a, b); });

	// Open input image with leptonica library
	Pix* image = pixRead(image_filename.c_str());
	std::cout << "w = " << image->w << " h = " << image->h << std::endl;

	float max_dim = std::max(image->w, image->h);
	auto max_size = 1690.f;
	auto scale = std::min(max_size / max_dim, 1.f);
	auto scaled_image = pixScale(image, scale, scale);
	std::cout << "w = " << scaled_image->w << " h = " << scaled_image->h << std::endl;

	try {
		Ocr_classifier classifier("fra", workers);
		classifier.set_classes(map);
		std::cout << classifier.classifiy(scaled_image) << std::endl;
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}
	return 0;
}
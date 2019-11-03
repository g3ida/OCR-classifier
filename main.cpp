#include "CLI11.hpp"
#include "json.hpp"
#include <leptonica/allheaders.h>
#include <iostream>

#include "ocr_classifier.hpp"
#include "east_detector.hpp"
#include "ocr_words_extractor.hpp"

auto load_config(std::string config_filename) {
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
	return map;
}

int main(int argc, char* argv[]) {
	CLI::App app{ "OCR Classifier" };

	//command options
	std::string dir_filename;
	std::string config_filename;
	std::string image_filename;
	std::string images_dir;
	std::string output_dir;
	std::string lang{ "eng" };
	float aquisition_rate{ 0.2f };
	int workers{ 2 };
	bool use_early_stopping = false;

	app.require_subcommand();
	auto extract_command = app.add_subcommand("extract", "extract relevant words from image files located in "
															"directories holding the desired class names.");
	auto predict_command = app.add_subcommand("predict", "predict the class of the image based on a json file "
															"containing relevant words for each class.");
	auto classify_command = app.add_subcommand("classify", "classify elements from a given directory into "
															"sub-folders denoting the corresponding classes.");
	
	//common options
	app.add_option("-l, --lang", lang, "OCR language");
	app.add_option("-w,--workers", workers, "workers threads")->check(CLI::PositiveNumber);
	//the extract command options
	extract_command->add_option("-d,--dir", dir_filename, "directory consisting of  sub-directories containing "
												       "images for each class")->required()->check(CLI::ExistingDirectory);
	extract_command->add_option("-o,--output", config_filename, "output json file")->required();
	extract_command->add_option("-r,--aquisition_rate", aquisition_rate, "words occuring belew that rate will get removed");
	//the predict command options
	predict_command->add_option("-c,--config", config_filename, "configuration file")->check(CLI::ExistingFile)->required();
	predict_command->add_option("-i,--image", image_filename, "image file")->check(CLI::ExistingFile)->required();
	predict_command->add_option("-e,--e", use_early_stopping, "enable early stopping");
	//the classf=ify command options
	classify_command->add_option("-d,--dir", images_dir, "directory containing images to classify")->check(CLI::ExistingDirectory)->required();
	classify_command->add_option("-c,--config", config_filename, "configuration file")->check(CLI::ExistingFile)->required();
	classify_command->add_option("-o,--output", output_dir, "output directory")->check(CLI::ExistingDirectory)->required();
	classify_command->add_option("-e,--e", use_early_stopping, "enable early stopping");


	CLI11_PARSE(app, argc, argv);

	if (!dir_filename.empty()) { //need to find a better way to do this
		//extract command
		Ocr_words_extractor extractor(dir_filename);
		extractor.process(lang, workers, aquisition_rate);
		extractor.save(config_filename);
	}
	else if(!output_dir.empty()) {
		//classify command
		auto map = load_config(config_filename);
		Ocr_classifier classifier(lang, workers, use_early_stopping);
		for (auto& el : map) {
			fs::create_directory(output_dir + '/' + el.first);
		}
		classifier.set_classes(std::move(map));
		for (const auto& entry : fs::directory_iterator(images_dir)) {
			if (is_supported_image_file_extension(entry.path().extension().string())) {
				Pix* image = pixRead(entry.path().string().c_str());
				auto scaled_image = scale_image(image, 1690.f);
				auto detected_class = classifier.classifiy(scaled_image);
				std::cout << detected_class << std::endl;
				auto dest = output_dir + '/' + detected_class + '/' +
					entry.path().filename().string();
				std::cout << "destination : " << dest << std::endl;
				pixWrite(dest.c_str() , image, image->informat);
				pixDestroy(&image);
				pixDestroy(&scaled_image);

			}
		}
	}
	else {
		auto map = load_config(config_filename);
		// Open input image with leptonica library
		Pix* image = pixRead(image_filename.c_str());
		auto scaled_image = scale_image(image, 1690.f);
		pixDestroy(&image);
		Ocr_classifier classifier(lang, workers, use_early_stopping);
		classifier.set_classes(map);
		std::cout << classifier.classifiy(scaled_image) << std::endl;
		pixDestroy(&scaled_image);
	}

	return 0;
}
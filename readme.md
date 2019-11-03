# OCR Classifier

OCR Classifier is a tool that helps to arrange images into folders based on optical character recognition of the text contained in the given image.

## Dependencies
This project requires C++17 compatible compiler.
In order to build this project, the following dependencies are required.
* Tesseract OCR
* Leptonica
* OpenCV

## Basic usage
The CLI for the OCR Classifier consist of 3 subcommands :

### `extract` command
Use this command to extract relevant text for each class you want to classify.

`OCR_Classifier extract --dir path/to/images/directory/ --output path/to/the/generated/config/file.json` 


### `predict` command
Use this command to predict the class for a single input image.

`OCR_Classifier predict --config path/to/config/file.json --image path/to/input/image.jpg` 


### `classify` command
Use this command to arrange images in an input directory into sub-folders for each class.

`OCR_Classifier classify --config path/to/config/file.json --dir path/to/images/directory/ --output path/to/the/output/directory/` 

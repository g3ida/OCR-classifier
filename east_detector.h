#pragma once
#include <string_view>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

class EAST_detector {
public :
	void load_model(std::string sv) {
		net_ = cv::dnn::readNet(sv);
	}

	std::vector<cv::RotatedRect> detect(cv::Mat frame) {
		cv::Mat blob;
		cv::dnn::blobFromImage(frame, blob, 1.0, cv::Size(inp_width_, inp_height_), 
			cv::Scalar(123.68, 116.78, 103.94), true, false);
		
		net_.setInput(blob);
		std::vector<cv::Mat> outs;
		std::vector<std::string> outNames(2);
		outNames[0] = "feature_fusion/Conv_7/Sigmoid";
		outNames[1] = "feature_fusion/concat_3";
		
		net_.forward(outs, outNames);

		cv::Mat scores = outs[0];
		cv::Mat geometry = outs[1];

		// Decode predicted bounding boxes.
		std::vector<cv::RotatedRect> boxes;
		std::vector<float> confidences;
		decode(scores, geometry, conf_threshold_, boxes, confidences);

		// Apply non-maximum suppression procedure.
		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
		
		cv::Point2f ratio((float)frame.cols / inp_width_, (float)frame.rows / inp_height_);
		
		std::vector<cv::RotatedRect> rects;
		for (size_t i = 0; i < indices.size(); ++i)
		{
			cv::RotatedRect& box = boxes[indices[i]];

			cv::Point2f vertices[4];
			box.points(vertices);
			for (int j = 0; j < 4; ++j)
			{
				vertices[j].x *= ratio.x;
				vertices[j].y *= ratio.y;
			}
			rects.emplace_back(cv::RotatedRect(vertices[0], vertices[1], vertices[2]));
		}
		return rects;
	}

private:
	cv::dnn::Net net_;
	int inp_width_ = 320;
	int inp_height_ = 320;
	float conf_threshold_ = 0.5f;
	float nms_threshold_ = 0.5f;


	void decode(const cv::Mat& scores, const cv::Mat& geometry, float scoreThresh,
			std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences) {
		
		detections.clear();
		CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
		CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
		CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

		const int height = scores.size[2];
		const int width = scores.size[3];
		for (int y = 0; y < height; ++y)
		{
			const float* scoresData = scores.ptr<float>(0, 0, y);
			const float* x0_data = geometry.ptr<float>(0, 0, y);
			const float* x1_data = geometry.ptr<float>(0, 1, y);
			const float* x2_data = geometry.ptr<float>(0, 2, y);
			const float* x3_data = geometry.ptr<float>(0, 3, y);
			const float* anglesData = geometry.ptr<float>(0, 4, y);
			for (int x = 0; x < width; ++x)
			{
				float score = scoresData[x];
				if (score < scoreThresh)
					continue;

				// Decode a prediction.
				// Multiple by 4 because feature maps are 4 time less than input image.
				float offsetX = x * 4.0f, offsetY = y * 4.0f;
				float angle = anglesData[x];
				float cosA = std::cos(angle);
				float sinA = std::sin(angle);
				float h = x0_data[x] + x2_data[x];
				float w = x1_data[x] + x3_data[x];

				cv::Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
					offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
				cv::Point2f p1 = cv::Point2f(-sinA * h, -cosA * h) + offset;
				cv::Point2f p3 = cv::Point2f(-cosA * w, sinA * w) + offset;
				cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
				detections.push_back(r);
				confidences.push_back(score);
			}
		}
	}

};
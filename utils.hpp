#pragma once

#include <math.h>
#include <algorithm>
#include <string.h>
#include <vector>
#include <string>
#include <string_view>
#include <leptonica/allheaders.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int levinstein_distance(std::string_view s1, std::string_view s2) {
	int i, j, l1, l2, t, track;
	int dist[50][50];

	//stores the lenght of strings s1 and s2
	l1 = s1.length();
	l2 = s2.length();
	for (i = 0; i <= l1; i++) {
		dist[0][i] = i;
	}
	for (j = 0; j <= l2; j++) {
		dist[j][0] = j;
	}
	for (j = 1; j <= l1; j++) {
		for (i = 1; i <= l2; i++) {
			if (s1[i-1] == s2[j-1]) {
				track = 0;
			}
			else {
				track = 1;
			}
			t = std::min((dist[i - 1][j] + 1), (dist[i][j - 1] + 1));
			dist[i][j] = std::min(t, (dist[i - 1][j - 1] + track));
		}
	}
	return dist[l2][l1];
}


PIX* mat_to_Pix(cv::Mat image)
{
	auto imagedata = static_cast<uchar*>(image.data);
	auto width = image.size().width;
	auto height = image.size().height;
	auto bytes_per_pixel = image.channels();
	auto bytes_per_line = image.step1();

	int bpp = bytes_per_pixel * 8;
	if (bpp == 0) bpp = 1;
	Pix* pix = pixCreate(width, height, bpp == 24 ? 32 : bpp);
	l_uint32* data = pixGetData(pix);
	int wpl = pixGetWpl(pix);
	switch (bpp) {
	case 1:
		for (int y = 0; y < height; ++y, data += wpl, imagedata += bytes_per_line) {
			for (int x = 0; x < width; ++x) {
				if (imagedata[x / 8] & (0x80 >> (x % 8)))
					CLEAR_DATA_BIT(data, x);
				else
					SET_DATA_BIT(data, x);
			}
		}
		break;

	case 8:
		// Greyscale just copies the bytes in the right order.
		for (int y = 0; y < height; ++y, data += wpl, imagedata += bytes_per_line) {
			for (int x = 0; x < width; ++x)
				SET_DATA_BYTE(data, x, imagedata[x]);
		}
		break;

	case 24:
		// Put the colors in the correct places in the line buffer.
		for (int y = 0; y < height; ++y, imagedata += bytes_per_line) {
			for (int x = 0; x < width; ++x, ++data) {
				SET_DATA_BYTE(data, COLOR_RED, imagedata[3 * x]);
				SET_DATA_BYTE(data, COLOR_GREEN, imagedata[3 * x + 1]);
				SET_DATA_BYTE(data, COLOR_BLUE, imagedata[3 * x + 2]);
			}
		}
		break;

	case 32:
		// Maintain byte order consistency across different endianness.
		for (int y = 0; y < height; ++y, imagedata += bytes_per_line, data += wpl) {
			for (int x = 0; x < width; ++x) {
				data[x] = (imagedata[x * 4] << 24) | (imagedata[x * 4 + 1] << 16) |
					(imagedata[x * 4 + 2] << 8) | imagedata[x * 4 + 3];
			}
		}
		break;

	default:
		return nullptr;
	}
	pixSetYRes(pix, 300);
	return pix;
}

cv::Mat pix8_to_Mat(PIX* pix8)
{
	cv::Mat mat(cv::Size(pix8->w, pix8->h), CV_8UC1);
	uint32_t* line = pix8->data;
	for (uint32_t y = 0; y < pix8->h; ++y) {
		for (uint32_t x = 0; x < pix8->w; ++x) {
			mat.at<uchar>(y, x) = GET_DATA_BYTE(line, x);
		}
		line += pix8->wpl;
	}
	return mat;
}

cv::Mat pix1_to_mat(Pix* pix8)
{
	//Convert the pix to the 8 depth
	PIX* pixd = nullptr;

	if (pix8->d == 1) {
		pixd = pixConvert1To8(NULL, pix8, 255, 0);
	}
	else if (pix8->d == 32) {
		pixd = pixConvert32To8(pix8, L_MS_TWO_BYTES, L_MS_BYTE);
	}
	else if (pixd == nullptr) {
		throw "error";
	}

	//pixWriteImpliedFormat("E://mm.jpg", pixd, 0, 0);

	cv::Mat mat(cv::Size(pixd->w, pixd->h), CV_8UC1);
	uint32_t* line = pixd->data;
	for (uint32_t y = 0; y < pixd->h; ++y) {
		for (uint32_t x = 0; x < pixd->w; ++x) {
			mat.at<uchar>(y, x) = GET_DATA_BYTE(line, x);
		}
		line += pixd->wpl;
	}
	pixDestroy(&pixd);
	return mat;
}

cv::Rect rect_add_margin(cv::Rect rec, int margin) {
	rec.x = std::max(rec.x - margin , 0);
	rec.y = std::max(rec.y - margin, 0);
	rec.width += margin;
	rec.height += margin;
	return rec;
}

bool is_supported_image_file_extension(std::string_view extension) {
	return (extension == ".bmp" || extension == ".tiff" || extension == ".png"
		|| extension == ".jpeg" || extension == ".jpg" || extension == ".pnm"
		|| extension == ".gif" || extension == ".webp");
}


std::vector<std::string> split_string(const std::string& s, char delimiter)
{
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter))
	{
		tokens.push_back(token);
	}
	return tokens;
}

PIX* scale_image(PIX* image, float max_size) {

	float max_dim = std::max(image->w, image->h);
	auto scale = std::min(max_size / max_dim, 1.f);
	auto scaled_image = pixScale(image, scale, scale);
	return scaled_image;
}
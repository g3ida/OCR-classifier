#pragma once

#include <math.h>
#include <algorithm>
#include <string.h>
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


Pix* mat8ToPix(cv::Mat* mat8)
{
	Pix* pixd = pixCreate(mat8->size().width, mat8->size().height, 8);
	for (int y = 0; y < mat8->rows; y++) {
		for (int x = 0; x < mat8->cols; x++) {
			pixSetPixel(pixd, x, y, (l_uint32)mat8->at<uchar>(y, x));
		}
	}
	return pixd;
}

cv::Mat pix8ToMat(Pix* pix8)
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

cv::Rect rect_add_margin(cv::Rect rec, int margin) {
	rec.x = std::max(rec.x - margin , 0);
	rec.y = std::max(rec.y - margin, 0);
	rec.width += margin;
	rec.height += margin;
	return rec;
}


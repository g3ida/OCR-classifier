#pragma once

#include <math.h>
#include <algorithm>
#include <string.h>
#include <string_view>

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

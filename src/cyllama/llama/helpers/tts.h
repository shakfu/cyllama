#define _USE_MATH_DEFINES // For M_PI on MSVC

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

enum outetts_version {
    OUTETTS_V0_2,
    OUTETTS_V0_3,
};

#define SQR(X)    ((X) * (X))
#define UNCUBE(x) x < 48 ? 0 : x < 115 ? 1 : (x - 35) / 40

int rgb2xterm256(int r, int g, int b);
std::string set_xterm256_foreground(int r, int g, int b);
bool save_wav16(const std::string & fname, const std::vector<float> & data, int sample_rate);

void fill_hann_window(int length, bool periodic, float * output);
void twiddle(float * real, float * imag, int k, int N);
void irfft(int n, const float * inp_cplx, float * out_real);
void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output);

std::string convert_less_than_thousand(int num);
std::string number_to_words(const std::string & number_str);
std::string replace_numbers_with_words(const std::string & input_text);
std::string process_text(const std::string & text, const outetts_version tts_version);

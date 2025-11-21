# distutils: language=c++

from libc.stdint cimport int64_t
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

cdef extern from "helpers/tts.h":

    ctypedef enum outetts_version:
        OUTETTS_V0_2
        OUTETTS_V0_3

    cdef int rgb2xterm256(int r, int g, int b)
    cdef std_string set_xterm256_foreground(int r, int g, int b)
    cdef bint save_wav16(const std_string & fname, const std_vector[float] & data, int sample_rate)

    cdef void fill_hann_window(int length, bint periodic, float * output)
    cdef void twiddle(float * real, float * imag, int k, int N)
    cdef void irfft(int n, const float * inp_cplx, float * out_real)
    cdef void fold(const std_vector[float] & data, int64_t n_out, int64_t n_win,
             int64_t n_hop, int64_t n_pad, std_vector[float] & output)

    cdef std_string convert_less_than_thousand(int num)
    cdef std_string number_to_words(const std_string & number_str)
    cdef std_string replace_numbers_with_words(const std_string & input_text)
    cdef std_string process_text(const std_string & text, const outetts_version tts_version)

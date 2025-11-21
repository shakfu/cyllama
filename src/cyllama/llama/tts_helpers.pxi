from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector as std_vector

cimport tts_helpers as tts

def rgb2xterm256(int r, int g, int b) -> int:
	return tts.rgb2xterm256(r, g, b)

def set_xterm256_foreground(int r, int g, int b) -> str:
	return tts.set_xterm256_foreground(r, g, b).decode()

def save_wav16_from_list(str fname, list[float] data, int sample_rate) -> bool:
	cdef std_vector[float] vec_data
	for val in data:
		vec_data.push_back(float(val))
	return tts.save_wav16(fname.encode(), vec_data, sample_rate)

def save_wav16(str fname, float[:] data, int sample_rate) -> bool:
	cdef std_vector[float] vec_data
	for val in data:
		vec_data.push_back(float(val))
	return tts.save_wav16(fname.encode(), vec_data, sample_rate)

def fill_hann_window(int length, bint periodic) -> list:
	cdef float* output = <float*>malloc(length * sizeof(float))
	if not output:
		raise MemoryError("Failed to allocate memory for Hann window")
	try:
		tts.fill_hann_window(length, periodic, output)
		result = [output[i] for i in range(length)]
		return result
	finally:
		free(output)

def twiddle_factors(float real, float imag, int k, int N) -> tuple:
	cdef float real_val = real
	cdef float imag_val = imag
	tts.twiddle(&real_val, &imag_val, k, N)
	return (real_val, imag_val)

def irfft(list inp_cplx) -> list:
	cdef int n = len(inp_cplx)
	cdef float* input_data = <float*>malloc(n * sizeof(float))
	cdef float* output_data = <float*>malloc(n * sizeof(float))
	if not input_data or not output_data:
		free(input_data)
		free(output_data)
		raise MemoryError("Failed to allocate memory for IRFFT")

	try:
		for i in range(n):
			input_data[i] = float(inp_cplx[i])

		tts.irfft(n, input_data, output_data)
		result = [output_data[i] for i in range(n)]
		return result
	finally:
		free(input_data)
		free(output_data)

def fold(list data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad) -> list:
	cdef std_vector[float] vec_data
	cdef std_vector[float] output

	for val in data:
		vec_data.push_back(float(val))

	tts.fold(vec_data, n_out, n_win, n_hop, n_pad, output)

	result = []
	for i in range(output.size()):
		result.append(output[i])
	return result

def convert_less_than_thousand(int num) -> str:
	return tts.convert_less_than_thousand(num).decode()

def number_to_words(str number_str) -> str:
	return tts.number_to_words(number_str.encode()).decode()

def replace_numbers_with_words(str input_text) -> str:
	return tts.replace_numbers_with_words(input_text.encode()).decode()

def process_text(str text, int tts_version) -> str:
	cdef tts.outetts_version version = <tts.outetts_version>tts_version
	return tts.process_text(text.encode(), version).decode()


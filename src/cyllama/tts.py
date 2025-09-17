#!/usr/bin/env python3
"""
TTS (Text-to-Speech) implementation equivalent to build/llama.cpp/tools/tts/tts.cpp

This module provides a Python implementation of the TTS example using the cyllama wrapper.
Based on OuteTTS model for high-quality text-to-speech synthesis.
"""

import sys
import argparse
import math
import struct
from typing import List, Optional, Dict, Tuple

from . import llama_cpp as cy


def save_wav16(filename: str, data: List[float], sample_rate: int) -> bool:
    """Save audio data as 16-bit WAV file"""
    return cy.save_wav16_from_list(filename, data, sample_rate)


def twiddle(k: int, N: int) -> Tuple[float, float]:
    """Compute twiddle factors for FFT"""
    return cy.twiddle_factors(1.0, 0.0, k, N)


def embd_to_audio(embd: List[float], n_codes: int, n_embd: int, n_threads: int = 4) -> List[float]:
    """Convert embeddings to audio using inverse spectrogram operations"""
    n_fft = 1280
    n_hop = 320
    n_win = 1280
    n_pad = (n_win - n_hop) // 2
    n_out = (n_codes - 1) * n_hop + n_win

    # Generate Hann window
    hann = cy.fill_hann_window(n_fft, True)

    n_spec = n_embd * n_codes

    # Transpose embeddings: E[k*n_codes + l] = embd[l*n_embd + k]
    E = [0.0] * n_spec
    for l in range(n_codes):
        for k in range(n_embd):
            E[k * n_codes + l] = embd[l * n_embd + k]

    # Convert to complex spectrogram
    S = [0.0] * n_spec
    for k in range(n_embd // 2):
        for l in range(n_codes):
            mag = math.exp(E[k * n_codes + l])
            phi = E[(k + n_embd // 2) * n_codes + l]

            # Clamp magnitude
            if mag > 1e2:
                mag = 1e2

            S[2 * (k * n_codes + l)] = mag * math.cos(phi)
            S[2 * (k * n_codes + l) + 1] = mag * math.sin(phi)

    # Transpose spectrogram for IRFFT
    ST = [0.0] * n_spec
    for l in range(n_codes):
        for k in range(n_embd // 2):
            ST[l * n_embd + 2 * k] = S[2 * (k * n_codes + l)]
            ST[l * n_embd + 2 * k + 1] = S[2 * (k * n_codes + l) + 1]

    # Apply IRFFT and windowing
    res = []
    hann2 = []

    for l in range(n_codes):
        # Apply IRFFT to get time-domain signal
        frame_spec = ST[l * n_embd:(l + 1) * n_embd]
        frame_audio = cy.irfft(frame_spec)  # Cython version only takes inp_cplx parameter

        # Apply window
        windowed_frame = [frame_audio[i] * hann[i] for i in range(n_fft)]
        squared_hann = [hann[i] * hann[i] for i in range(n_fft)]

        res.extend(windowed_frame)
        hann2.extend(squared_hann)

    # Overlap-add using fold operation
    audio = cy.fold(res, n_out, n_win, n_hop, n_pad)
    env = cy.fold(hann2, n_out, n_win, n_hop, n_pad)

    # Normalize by envelope
    for i in range(len(audio)):
        if env[i] > 0:
            audio[i] /= env[i]

    return audio



def process_text(text: str, tts_version: str = "0.2") -> str:
    """Process text for TTS input"""
    # Convert version string to int for Cython function
    version_int = 0 if tts_version == "0.2" else 1  # OUTETTS_V0_2 = 0, OUTETTS_V0_3 = 1
    return cy.process_text(text, version_int)


def prepare_guide_tokens(vocab, text: str, tts_version: str = "0.2") -> List[int]:
    """Prepare guide tokens to prevent hallucinations"""
    delimiter = "<|space|>" if tts_version == "0.3" else "<|text_sep|>"

    result = []

    # First token is always a newline
    newline_tokens = vocab.tokenize("\n", False, True)
    if newline_tokens:
        result.append(newline_tokens[0])

    # Split by delimiter and tokenize each word
    words = text.split(delimiter)

    for word in words:
        if word:
            word_tokens = vocab.tokenize(word, False, True)
            if word_tokens:
                result.append(word_tokens[0])

    return result


class TTSGenerator:
    """Text-to-Speech generator using OuteTTS models"""

    def __init__(self,
                 ttc_model_path: str,  # text-to-codes model
                 cts_model_path: str,  # codes-to-speech model
                 n_ctx: int = 8192,
                 n_batch: int = 8192,
                 ngl: int = 99,
                 n_predict: int = 4096,
                 speaker_file: Optional[str] = None,
                 use_guide_tokens: bool = True):
        """Initialize TTS with models and parameters"""

        # Load dynamic backends
        cy.ggml_backend_load_all()

        # Initialize text-to-codes model
        model_params = cy.LlamaModelParams()
        model_params.n_gpu_layers = ngl
        self.model_ttc = cy.LlamaModel(ttc_model_path, model_params)
        self.vocab = self.model_ttc.get_vocab()

        # Initialize text-to-codes context
        ctx_params = cy.LlamaContextParams()
        ctx_params.n_ctx = n_ctx
        ctx_params.n_batch = n_batch
        self.context_ttc = cy.LlamaContext(self.model_ttc, ctx_params)

        # Initialize codes-to-speech model
        model_params_cts = cy.LlamaModelParams()
        model_params_cts.n_gpu_layers = ngl
        self.model_cts = cy.LlamaModel(cts_model_path, model_params_cts)

        # Initialize codes-to-speech context (embedding mode)
        ctx_params_cts = cy.LlamaContextParams()
        ctx_params_cts.n_ctx = n_ctx
        ctx_params_cts.n_batch = n_batch
        ctx_params_cts.n_ubatch = n_batch
        self.context_cts = cy.LlamaContext(self.model_cts, ctx_params_cts)

        # Set embedding mode for codes-to-speech
        self.context_cts.set_embeddings_mode(True)

        # Initialize sampler for text-to-codes generation
        sampler_params = cy.LlamaSamplerChainParams()
        self.sampler = cy.LlamaSampler(sampler_params)
        self.sampler.add_top_k(4)
        self.sampler.add_temp(0.8)
        self.sampler.add_dist(1337)  # Use fixed seed

        # Store parameters
        self.n_predict = n_predict
        self.use_guide_tokens = use_guide_tokens
        self.speaker_file = speaker_file

        # Determine TTS version
        chat_template = self.model_ttc.get_default_chat_template()
        if chat_template and "outetts-0.3" in chat_template:
            self.tts_version = "0.3"
        else:
            self.tts_version = "0.2"

        # Load speaker data if provided
        if speaker_file:
            self.load_speaker(speaker_file)
        else:
            self.setup_default_speaker()

    def load_speaker(self, speaker_file: str):
        """Load speaker profile from JSON file"""
        import json
        try:
            with open(speaker_file, 'r') as f:
                speaker_data = json.load(f)

            # Extract version if available
            if 'version' in speaker_data:
                self.tts_version = speaker_data['version']

            # Build audio text and data from speaker
            self.audio_text = self.audio_text_from_speaker(speaker_data)
            self.audio_data = self.audio_data_from_speaker(speaker_data)

        except Exception as e:
            print(f"Error loading speaker file: {e}", file=sys.stderr)
            self.setup_default_speaker()

    def setup_default_speaker(self):
        """Setup default speaker profile"""
        # Default speaker profile based on OuteTTS en_male_1 - matches C++ version exactly
        separator = "<|space|>" if self.tts_version == "0.3" else "<|text_sep|>"

        self.audio_text = f"<|text_start|>the{separator}overall{separator}package{separator}from{separator}just{separator}two{separator}people{separator}is{separator}pretty{separator}remarkable{separator}sure{separator}i{separator}have{separator}some{separator}critiques{separator}about{separator}some{separator}of{separator}the{separator}gameplay{separator}aspects{separator}but{separator}its{separator}still{separator}really{separator}enjoyable{separator}and{separator}it{separator}looks{separator}lovely{separator}"

        # Full audio data template matching C++ version exactly
        self.audio_data = """<|audio_start|>
the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>
overall<|t_0.36|><|code_start|><|127|><|201|><|191|><|774|><|700|><|532|><|1056|><|557|><|798|><|298|><|1741|><|747|><|1662|><|1617|><|1702|><|1527|><|368|><|1588|><|1049|><|1008|><|1625|><|747|><|1576|><|728|><|1019|><|1696|><|1765|><|code_end|>
package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>
from<|t_0.19|><|code_start|><|604|><|782|><|1682|><|872|><|1532|><|1600|><|1036|><|1761|><|647|><|1554|><|1371|><|653|><|1595|><|950|><|code_end|>
just<|t_0.25|><|code_start|><|1782|><|1670|><|317|><|786|><|1748|><|631|><|599|><|1155|><|1364|><|1524|><|36|><|1591|><|889|><|1535|><|541|><|440|><|1532|><|50|><|870|><|code_end|>
two<|t_0.24|><|code_start|><|1681|><|1510|><|673|><|799|><|805|><|1342|><|330|><|519|><|62|><|640|><|1138|><|565|><|1552|><|1497|><|1552|><|572|><|1715|><|1732|><|code_end|>
people<|t_0.39|><|code_start|><|593|><|274|><|136|><|740|><|691|><|633|><|1484|><|1061|><|1138|><|1485|><|344|><|428|><|397|><|1562|><|645|><|917|><|1035|><|1449|><|1669|><|487|><|442|><|1484|><|1329|><|1832|><|1704|><|600|><|761|><|653|><|269|><|code_end|>
is<|t_0.16|><|code_start|><|566|><|583|><|1755|><|646|><|1337|><|709|><|802|><|1008|><|485|><|1583|><|652|><|10|><|code_end|>
pretty<|t_0.32|><|code_start|><|1818|><|1747|><|692|><|733|><|1010|><|534|><|406|><|1697|><|1053|><|1521|><|1355|><|1274|><|816|><|1398|><|211|><|1218|><|817|><|1472|><|1703|><|686|><|13|><|822|><|445|><|1068|><|code_end|>
remarkable<|t_0.68|><|code_start|><|230|><|1048|><|1705|><|355|><|706|><|1149|><|1535|><|1787|><|1356|><|1396|><|835|><|1583|><|486|><|1249|><|286|><|937|><|1076|><|1150|><|614|><|42|><|1058|><|705|><|681|><|798|><|934|><|490|><|514|><|1399|><|572|><|1446|><|1703|><|1346|><|1040|><|1426|><|1304|><|664|><|171|><|1530|><|625|><|64|><|1708|><|1830|><|1030|><|443|><|1509|><|1063|><|1605|><|1785|><|721|><|1440|><|923|><|code_end|>
sure<|t_0.36|><|code_start|><|792|><|1780|><|923|><|1640|><|265|><|261|><|1525|><|567|><|1491|><|1250|><|1730|><|362|><|919|><|1766|><|543|><|1|><|333|><|113|><|970|><|252|><|1606|><|133|><|302|><|1810|><|1046|><|1190|><|1675|><|code_end|>
i<|t_0.08|><|code_start|><|123|><|439|><|1074|><|705|><|1799|><|637|><|code_end|>
have<|t_0.16|><|code_start|><|1509|><|599|><|518|><|1170|><|552|><|1029|><|1267|><|864|><|419|><|143|><|1061|><|0|><|code_end|>
some<|t_0.16|><|code_start|><|619|><|400|><|1270|><|62|><|1370|><|1832|><|917|><|1661|><|167|><|269|><|1366|><|1508|><|code_end|>
critiques<|t_0.60|><|code_start|><|559|><|584|><|1163|><|1129|><|1313|><|1728|><|721|><|1146|><|1093|><|577|><|928|><|27|><|630|><|1080|><|1346|><|1337|><|320|><|1382|><|1175|><|1682|><|1556|><|990|><|1683|><|860|><|1721|><|110|><|786|><|376|><|1085|><|756|><|1523|><|234|><|1334|><|1506|><|1578|><|659|><|612|><|1108|><|1466|><|1647|><|308|><|1470|><|746|><|556|><|1061|><|code_end|>
about<|t_0.29|><|code_start|><|26|><|1649|><|545|><|1367|><|1263|><|1728|><|450|><|859|><|1434|><|497|><|1220|><|1285|><|179|><|755|><|1154|><|779|><|179|><|1229|><|1213|><|922|><|1774|><|1408|><|code_end|>
some<|t_0.23|><|code_start|><|986|><|28|><|1649|><|778|><|858|><|1519|><|1|><|18|><|26|><|1042|><|1174|><|1309|><|1499|><|1712|><|1692|><|1516|><|1574|><|code_end|>
of<|t_0.07|><|code_start|><|197|><|716|><|1039|><|1662|><|64|><|code_end|>
the<|t_0.08|><|code_start|><|1811|><|1568|><|569|><|886|><|1025|><|1374|><|code_end|>
gameplay<|t_0.48|><|code_start|><|1269|><|1092|><|933|><|1362|><|1762|><|1700|><|1675|><|215|><|781|><|1086|><|461|><|838|><|1022|><|759|><|649|><|1416|><|1004|><|551|><|909|><|787|><|343|><|830|><|1391|><|1040|><|1622|><|1779|><|1360|><|1231|><|1187|><|1317|><|76|><|997|><|989|><|978|><|737|><|189|><|code_end|>
aspects<|t_0.56|><|code_start|><|1423|><|797|><|1316|><|1222|><|147|><|719|><|1347|><|386|><|1390|><|1558|><|154|><|440|><|634|><|592|><|1097|><|1718|><|712|><|763|><|1118|><|1721|><|1311|><|868|><|580|><|362|><|1435|><|868|><|247|><|221|><|886|><|1145|><|1274|><|1284|><|457|><|1043|><|1459|><|1818|><|62|><|599|><|1035|><|62|><|1649|><|778|><|code_end|>
but<|t_0.20|><|code_start|><|780|><|1825|><|1681|><|1007|><|861|><|710|><|702|><|939|><|1669|><|1491|><|613|><|1739|><|823|><|1469|><|648|><|code_end|>
its<|t_0.09|><|code_start|><|92|><|688|><|1623|><|962|><|1670|><|527|><|599|><|code_end|>
still<|t_0.27|><|code_start|><|636|><|10|><|1217|><|344|><|713|><|957|><|823|><|154|><|1649|><|1286|><|508|><|214|><|1760|><|1250|><|456|><|1352|><|1368|><|921|><|615|><|5|><|code_end|>
really<|t_0.36|><|code_start|><|55|><|420|><|1008|><|1659|><|27|><|644|><|1266|><|617|><|761|><|1712|><|109|><|1465|><|1587|><|503|><|1541|><|619|><|197|><|1019|><|817|><|269|><|377|><|362|><|1381|><|507|><|1488|><|4|><|1695|><|code_end|>
enjoyable<|t_0.49|><|code_start|><|678|><|501|><|864|><|319|><|288|><|1472|><|1341|><|686|><|562|><|1463|><|619|><|1563|><|471|><|911|><|730|><|1811|><|1006|><|520|><|861|><|1274|><|125|><|1431|><|638|><|621|><|153|><|876|><|1770|><|437|><|987|><|1653|><|1109|><|898|><|1285|><|80|><|593|><|1709|><|843|><|code_end|>
and<|t_0.15|><|code_start|><|1285|><|987|><|303|><|1037|><|730|><|1164|><|502|><|120|><|1737|><|1655|><|1318|><|code_end|>
it<|t_0.09|><|code_start|><|848|><|1366|><|395|><|1601|><|1513|><|593|><|1302|><|code_end|>
looks<|t_0.27|><|code_start|><|1281|><|1266|><|1755|><|572|><|248|><|1751|><|1257|><|695|><|1380|><|457|><|659|><|585|><|1315|><|1105|><|1776|><|736|><|24|><|736|><|654|><|1027|><|code_end|>
lovely<|t_0.56|><|code_start|><|634|><|596|><|1766|><|1556|><|1306|><|1285|><|1481|><|1721|><|1123|><|438|><|1246|><|1251|><|795|><|659|><|1381|><|1658|><|217|><|1772|><|562|><|952|><|107|><|1129|><|1112|><|467|><|550|><|1079|><|840|><|1615|><|1469|><|1380|><|168|><|917|><|836|><|1827|><|437|><|583|><|67|><|595|><|1087|><|1646|><|1493|><|1677|><|code_end|>
"""
        if self.tts_version == "0.3":
            # Convert format for v0.3
            self.audio_data = self.audio_data.replace("<|code_start|>", "").replace("<|code_end|>", "<|space|>")

    def audio_text_from_speaker(self, speaker_data: Dict) -> str:
        """Extract audio text from speaker data"""
        audio_text = "<|text_start|>"
        separator = "<|space|>" if self.tts_version == "0.3" else "<|text_sep|>"

        if 'words' in speaker_data:
            for word in speaker_data['words']:
                audio_text += word['word'] + separator

        return audio_text

    def audio_data_from_speaker(self, speaker_data: Dict) -> str:
        """Extract audio data from speaker data"""
        audio_data = "<|audio_start|>\n"

        if 'words' in speaker_data:
            code_start = "" if self.tts_version == "0.3" else "<|code_start|>"
            code_end = "<|space|>" if self.tts_version == "0.3" else "<|code_end|>"

            for word in speaker_data['words']:
                word_text = word['word']
                duration = word['duration']
                codes = word['codes']

                entry = f"{word_text}<|t_{duration:.2f}|>{code_start}"
                for code in codes:
                    entry += f"<|{code}|>"
                entry += code_end + "\n"
                audio_data += entry

        return audio_data

    def generate_codes(self, text: str) -> List[int]:
        """Generate audio codes from text"""
        print(f"Processing text: '{text}'")

        # Process input text
        processed_text = process_text(text, self.tts_version)
        print(f"Processed text: '{processed_text}'")

        # Prepare guide tokens if enabled
        guide_tokens = []
        if self.use_guide_tokens:
            guide_tokens = prepare_guide_tokens(self.vocab, processed_text, self.tts_version)
            print(f"Using {len(guide_tokens)} guide tokens to ensure accurate pronunciation")

        # Build prompt
        prompt_tokens = []

        # Start with system prompt
        start_tokens = self.vocab.tokenize("<|im_start|>\n", True, True)
        prompt_tokens.extend(start_tokens)

        # Add audio text
        audio_text_tokens = self.vocab.tokenize(self.audio_text, False, True)
        prompt_tokens.extend(audio_text_tokens)

        # Add processed user text
        text_tokens = self.vocab.tokenize(processed_text, False, True)
        prompt_tokens.extend(text_tokens)

        # Add text end marker
        end_tokens = self.vocab.tokenize("<|text_end|>\n", False, True)
        prompt_tokens.extend(end_tokens)

        # Add audio data (speaker profile) - this provides the voice template
        # The model needs to see the full template pattern to understand the format
        if self.speaker_file:
            audio_data_tokens = self.vocab.tokenize(self.audio_data, False, True)
            prompt_tokens.extend(audio_data_tokens)
        else:
            # Use the full audio_data template - the model needs this pattern
            audio_data_tokens = self.vocab.tokenize(self.audio_data, False, True)
            prompt_tokens.extend(audio_data_tokens)

        print(f"Prompt size: {len(prompt_tokens)} tokens")

        # Create batch for initial prompt
        batch = cy.LlamaBatch(n_tokens=max(len(prompt_tokens), 1), embd=0, n_seq_max=1)

        # Add prompt tokens to batch
        batch.add_sequence(prompt_tokens, 0, False)

        # Mark last token for logits
        batch.set_last_logits_to_true()

        # Decode initial prompt
        ret = self.context_ttc.decode(batch)
        if ret != 0:
            print(f"Warning: initial decode returned {ret}")

        # Generation loop - aligned with C++ implementation
        codes = []
        n_past = len(prompt_tokens)
        n_decode = 0
        guide_tokens_copy = guide_tokens.copy()  # Make a copy to consume
        next_token_uses_guide_token = True

        # Generate until n_predict or EOS like C++ version
        audio_codes_generated = 0

        while n_decode <= self.n_predict:
            if n_past >= self.context_ttc.n_ctx - 1:
                print("Context size exceeded")
                break

            # Sample next token
            try:
                new_token_id = self.sampler.sample(self.context_ttc, -1)
            except Exception as e:
                print(f"Sampling failed: {e}")
                break

            # Guide tokens help prevent hallucinations by forcing the TTS to use the correct word
            # This logic matches the C++ implementation (lines 884-889)
            if (guide_tokens_copy and next_token_uses_guide_token and
                not self.vocab.is_control(new_token_id) and
                not self.vocab.is_eog(new_token_id)):
                guide_token = guide_tokens_copy.pop(0)  # Remove first token
                new_token_id = guide_token  # Ensure correct word fragment is used

            # This is the token id that always precedes a new word (matches C++ line 892)
            next_token_uses_guide_token = (new_token_id == 198)  # newline token

            # Accept the sampled token (matches C++ line 894)
            self.sampler.accept(new_token_id)

            codes.append(new_token_id)

            # Count audio codes as we generate them
            if 151672 <= new_token_id <= 155772:
                audio_codes_generated += 1

            # Check for end of generation (matches C++ lines 901-917)
            if self.vocab.is_eog(new_token_id) or n_decode == self.n_predict:
                reason = "eos" if self.vocab.is_eog(new_token_id) else "n_predict"
                print(f"Stopped at {reason} after {n_decode+1} tokens")
                break


            n_decode += 1
            n_past += 1

            # Create batch for next token
            batch.set_batch([new_token_id], n_past-1, True)  # Use n_past-1 for position

            # Decode next token
            ret = self.context_ttc.decode(batch)
            if ret != 0:
                print(f"Warning: decode returned {ret}")
                break

        # Batch will be automatically freed when it goes out of scope

        print(f"Generated {len(codes)} tokens")

        # Debug: Show generated text
        try:
            generated_text = self.vocab.detokenize(codes)
            print(f"Generated text preview: {generated_text[:200]}...")
        except:
            pass

        # Filter to audio codes only (token range 151672-155772) - matches C++ line 1003
        audio_codes = [t for t in codes if 151672 <= t <= 155772]
        print(f"Filtered to {len(audio_codes)} audio codes for vocoder")

        # Debug: Show filtered audio text
        try:
            audio_text = self.vocab.detokenize(audio_codes)
            print(f"Audio codes text: {audio_text[:100]}...")
        except:
            pass

        # Adjust token values for vocoder input (matches C++ lines 1011-1013)
        adjusted_codes = [t - 151672 for t in audio_codes]

        return adjusted_codes

    def codes_to_audio(self, codes: List[int]) -> List[float]:
        """Convert audio codes to audio samples"""
        n_codes = len(codes)

        # Create batch for codes
        batch = cy.LlamaBatch(n_tokens=n_codes, embd=0, n_seq_max=1)

        batch.add_sequence(codes, 0, True)

        # Encode codes to get embeddings
        try:
            self.context_cts.encode(batch)
        except Exception as e:
            print(f"Error encoding batch: {e}")
            return []

        # Get embeddings for all tokens at once
        n_embd = self.model_cts.n_embd

        try:
            # Try to get all embeddings at once (should be n_codes * n_embd floats)
            embeddings = self.context_cts.get_embeddings()
            # If we got fewer embeddings than expected, collect them individually (slower fallback)
            if len(embeddings) < n_codes * n_embd:
                print(f"Collecting embeddings for {n_codes} audio codes...")
                embeddings = []
                for i in range(n_codes):
                    if i % 100 == 0 and i > 0:
                        print(f"Processing audio code {i}/{n_codes}")
                    token_embeddings = self.context_cts.get_embeddings_ith(i)
                    embeddings.extend(token_embeddings)
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []

        if not embeddings:
            print("Error: no embeddings returned")
            return []

        print(f"Converting {len(embeddings)} embeddings to audio waveform...")

        # Convert embeddings to audio
        try:
            audio = embd_to_audio(embeddings, n_codes, n_embd, 4)
        except Exception as e:
            print(f"Error in embd_to_audio: {e}")
            import traceback
            traceback.print_exc()
            return []

        # Batch will be automatically freed when it goes out of scope

        # Zero out first 0.25 seconds to remove artifacts
        sample_rate = 24000
        zero_samples = sample_rate // 4
        for i in range(min(zero_samples, len(audio))):
            audio[i] = 0.0

        return audio

    def generate(self, text: str, output_file: str = "output.wav") -> bool:
        """Generate speech from text and save to file"""
        try:
            print(f"Generating speech for: '{text}'")

            # Generate audio codes from text
            codes = self.generate_codes(text)
            if not codes:
                print("Error: no codes generated")
                return False

            # Convert codes to audio
            audio = self.codes_to_audio(codes)
            if not audio:
                print("Error: no audio generated")
                return False

            # Save to WAV file
            sample_rate = 24000
            success = save_wav16(output_file, audio, sample_rate)

            if success:
                print(f"Audio written to file '{output_file}'")
                print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

            return success

        except Exception as e:
            print(f"Error generating speech: {e}", file=sys.stderr)
            return False


def main():
    """Main entry point"""
    cy.disable_logging()

    parser = argparse.ArgumentParser(description="Text-to-Speech using cyllama")
    parser.add_argument("-m", "--model", required=True, help="Path to text-to-codes model file")
    parser.add_argument("-mv", "--vocoder-model", required=True, help="Path to codes-to-speech model file")
    parser.add_argument("-p", "--prompt", required=True, help="Text to synthesize")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV file")
    parser.add_argument("-c", "--context", type=int, default=8192, help="Context size")
    parser.add_argument("-b", "--batch", type=int, default=8192, help="Batch size")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99, help="Number of GPU layers")
    parser.add_argument("-n", "--n-predict", type=int, default=4096, help="Number of tokens to predict")
    parser.add_argument("--speaker-file", help="Speaker profile JSON file")
    parser.add_argument("--use-guide-tokens", action="store_true", default=True, help="Use guide tokens to prevent hallucinations")
    parser.add_argument("--no-guide-tokens", action="store_true", help="Disable guide tokens")

    args = parser.parse_args()

    try:
        # Handle guide tokens setting
        use_guide_tokens = args.use_guide_tokens and not args.no_guide_tokens

        tts = TTSGenerator(
            ttc_model_path=args.model,
            cts_model_path=args.vocoder_model,
            n_ctx=args.context,
            n_batch=args.batch,
            ngl=args.n_gpu_layers,
            n_predict=args.n_predict,
            speaker_file=args.speaker_file,
            use_guide_tokens=use_guide_tokens
        )

        success = tts.generate(args.prompt, args.output)
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
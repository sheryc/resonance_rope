from enum import Enum


class EncodingType(Enum):
    POSITION_ENCODING_REL_T5_BIAS = "t5_relative_bias"
    POSITION_ENCODING_REL_TRANSFORMER_XL = "transformer_xl_relative_encoding"
    POSITION_ENCODING_ROTARY = "rotary"
    POSITION_ENCODING_ROTARY_RERUN = "rotary_rerun"
    POSITION_ENCODING_ROTARY_NEW = "new_rotary"
    POSITION_ENCODING_ROTARY_HF = "rotary_hf"
    POSITION_ENCODING_ROTARY_SCALED_LINEAR = "rotary_scaled_linear"
    POSITION_ENCODING_ROTARY_SCALED_NTK = "rotary_scaled_ntk"
    POSITION_ENCODING_ROTARY_SCALED_YARN = "rotary_scaled_yarn"
    POSITION_ENCODING_ROTARY_RESONANCE = "rotary_resonance"
    POSITION_ENCODING_ROTARY_RESONANCE_LINEAR = "rotary_resonance_linear"
    POSITION_ENCODING_ROTARY_RESONANCE_NTK = "rotary_resonance_ntk"
    POSITION_ENCODING_ROTARY_RESONANCE_YARN = "rotary_resonance_yarn"
    POSITION_ENCODING_ABS_LEARNED = "abs_learned"
    POSITION_ENCODING_ABS_SINUSOID = "abs_sinusoid"
    POSITION_ENCODING_ALiBi = "alibi"
    POSITION_ENCODING_ALiBi_LEARNED = "alibi_learned"
    POSITION_ENCODING_NONE = "none"


if __name__ == '__main__':
    print(EncodingType('rotary'))

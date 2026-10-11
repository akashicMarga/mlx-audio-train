"""
Compiled codec decoder with length buckets (Part 4, §5.3).

The 12 Hz decoder (RVQ dequant → causal conv → causal transformer → causal
transposed-conv upsampling) runs once per utterance on codes whose length
differs every time, so compiling on raw lengths would retrace per utterance.
Instead codes are right-padded with zeros to a multiple of `bucket` frames,
decoded by one compiled function per bucket, and the waveform is trimmed back
to T × 1920 samples.

That is only valid because every stage is causal: causal convs pad on the left,
the transformer uses a causal mask, CausalTransposeConv1d trims its right edge,
and the RVQ / norms / SnakeBeta act per frame — appended frames cannot change
earlier samples. Utterances longer than one stock chunk (300 frames) fall back
to the stock chunked path. Valid lengths are computed from the original codes,
exactly as Qwen3TTSSpeechTokenizer.decode does.
"""

import mlx.core as mx

STOCK_CHUNK = 300


def install(model, bucket: int = 16):
    tok = model.speech_tokenizer
    dec = tok.decoder
    stock_decode = tok.decode
    compiled = {}                      # padded length -> compiled fn (MLX also caches per shape)

    def fn_for(n):
        f = compiled.get(n)
        if f is None:
            f = mx.compile(lambda codes: dec(codes), inputs=[dec.state])
            compiled[n] = f
        return f

    def decode(audio_codes):
        # audio_codes: [batch, time, num_quantizers]
        t = audio_codes.shape[1]
        padded = -(-t // bucket) * bucket
        if padded > STOCK_CHUNK:
            return stock_decode(audio_codes)
        codes = mx.transpose(audio_codes, (0, 2, 1))
        if padded > t:
            codes = mx.pad(codes, [(0, 0), (0, 0), (0, padded - t)])
        wav = fn_for(padded)(codes)[..., : t * dec.total_upsample].squeeze(1)
        audio_lengths = (audio_codes[..., 0] > 0).sum(axis=1) * tok.decode_upsample_rate
        return wav, audio_lengths

    tok.decode = decode
    model._bench_decoder_buckets = compiled
    return model

# Audio Chunking for Large Files

Parakeet.js now supports chunked audio processing to handle large audio files that may cause memory issues when processed in a single pass.

## Problem

For long audio files (e.g., >5 minutes), processing the entire audio at once can lead to:
- High memory usage during encoding
- Browser memory constraints
- Out-of-memory errors

## Solution

The chunking algorithm splits large audio files into overlapping segments, processes each independently, and merges the results. This approach is based on NVIDIA NeMo's buffered inference technique.

## Usage

### Basic Chunking

To enable chunking, simply add the `chunkLengthSecs` parameter to your `transcribe()` call:

```javascript
const result = await model.transcribe(audio, 16000, {
  chunkLengthSecs: 10,  // Process in 10-second chunks
});
```

### Advanced Options

For more control, you can specify the buffer size (which includes padding for context):

```javascript
const result = await model.transcribe(audio, 16000, {
  chunkLengthSecs: 10,      // Core chunk size
  bufferLengthSecs: 15,     // Total buffer (chunk + padding)
  returnTimestamps: true,   // Get word timings
  returnConfidences: true,  // Get confidence scores
});
```

### How It Works

1. **Chunking**: Audio is split into overlapping segments
   - Each chunk is `chunkLengthSecs` seconds long
   - Chunks overlap by `(bufferLengthSecs - chunkLengthSecs) / 2` seconds on each side
   - Example: With 10s chunks and 15s buffer, each chunk overlaps by 2.5s on each side

2. **Processing**: Each chunk is processed independently
   - Feature extraction (log-mel spectrogram)
   - Encoder inference
   - Decoder inference (frame-by-frame)

3. **Merging**: Results from overlapping regions are merged
   - Uses "middle token" algorithm: keeps tokens closest to chunk center
   - Ensures smooth transitions between chunks
   - Maintains timestamp accuracy

### Parameter Guidelines

**Chunk Length** (`chunkLengthSecs`):
- **Short files (<30s)**: Disable chunking (omit parameter)
- **Medium files (30s-2min)**: 15-20 seconds
- **Long files (>2min)**: 10-15 seconds
- **Very long files (>10min)**: 8-10 seconds

**Buffer Length** (`bufferLengthSecs`):
- Should be 1.5-2x the chunk length
- Larger buffer = better context = better accuracy
- Larger buffer = more memory per chunk
- Default: 1.5x chunk length

### Performance

**Memory Savings**:
- Without chunking: Memory scales linearly with audio length
- With chunking: Memory is constant (depends only on chunk size)

**Speed**:
- Chunking adds ~2-5% overhead for merging
- Total processing time is similar to non-chunked
- Can enable future optimizations (parallel chunk processing)

### Examples

#### Example 1: Transcribe a 5-minute audio file

```javascript
// Without chunking (may cause memory issues)
const result = await model.transcribe(longAudio, 16000);

// With chunking (recommended)
const result = await model.transcribe(longAudio, 16000, {
  chunkLengthSecs: 10,
});

console.log(result.utterance_text);
console.log(`Processed in ${result.metrics.num_chunks} chunks`);
```

#### Example 2: Get detailed results with chunking

```javascript
const result = await model.transcribe(longAudio, 16000, {
  chunkLengthSecs: 12,
  bufferLengthSecs: 18,
  returnTimestamps: true,
  returnConfidences: true,
});

// Access word-level results
result.words.forEach(word => {
  console.log(`${word.text}: ${word.start_time}s - ${word.end_time}s (confidence: ${word.confidence})`);
});
```

#### Example 3: Debug chunking

```javascript
const result = await model.transcribe(longAudio, 16000, {
  chunkLengthSecs: 10,
  debug: true,  // Enable debug logging
});
```

This will log:
```
[Chunked] Processing 180.50s audio in chunks of 10s (buffer: 15s)
[Chunked] Will process 19 chunks
[Chunked] Processing chunk 1/19 (samples 0-160000)
[Chunked] Chunk 1 completed in 234.5ms, got 45 tokens
...
[Chunked] Merged 872 tokens from chunks into 845 final tokens
```

## Algorithm Details

### Based on NeMo's Buffered Inference

The implementation is inspired by NVIDIA NeMo's buffered RNNT inference:
- [speech_to_text_buffered_infer_rnnt.py](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py)

### Merge Strategy

The "middle token" algorithm is used to merge overlapping predictions:

1. Each token is associated with its position in the original audio
2. For positions that appear in multiple chunks, keep the token from the chunk where that position is closest to the center
3. This ensures tokens are predicted with maximum context available

### Timestamp Preservation

- Timestamps are calculated relative to the original audio position
- The chunking process is transparent to timestamp accuracy
- Word and token boundaries are preserved across chunk boundaries

## Limitations

1. **Decoder State**: Currently, decoder LSTM state is reset between chunks
   - This may cause minor accuracy differences at chunk boundaries
   - Future versions may support stateful decoding across chunks

2. **Sequential Processing**: Chunks are processed sequentially
   - Future versions may support parallel processing for better speed

3. **Memory vs. Accuracy Trade-off**:
   - Smaller chunks = less memory but more chunk boundaries
   - Larger buffer = better accuracy but more memory per chunk

## Troubleshooting

**Problem**: Still getting out-of-memory errors with chunking enabled

**Solutions**:
- Reduce `chunkLengthSecs` (try 8 or 6 seconds)
- Reduce `bufferLengthSecs` (but keep it > chunkLengthSecs)
- Use WASM backend instead of WebGPU (slower but uses less GPU memory)

**Problem**: Accuracy is worse with chunking

**Solutions**:
- Increase `bufferLengthSecs` for more context
- Increase `chunkLengthSecs` if memory allows
- Ensure buffer is at least 1.5x the chunk length

**Problem**: Transcription is slower with chunking

**Solutions**:
- This is expected for small files; only use chunking for files >30s
- The overhead is mostly from processing overlapping regions
- Benefits come from avoiding OOM errors, not from speed

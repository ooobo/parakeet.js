/**
 * Audio chunking utilities for handling large audio files.
 *
 * Based on the buffered inference approach from NVIDIA NeMo:
 * https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py
 *
 * This module provides utilities to:
 * 1. Split long audio into overlapping chunks
 * 2. Merge predictions from overlapping regions
 * 3. Manage encoder features for chunked processing
 */

/**
 * Calculate chunk parameters for audio processing
 * @param {number} audioDurationSecs - Total audio duration in seconds
 * @param {number} chunkLengthSecs - Desired chunk length in seconds
 * @param {number} bufferLengthSecs - Total buffer (chunk + padding) in seconds
 * @param {number} sampleRate - Audio sample rate (usually 16000)
 * @returns {Object} Chunk configuration
 */
export function calculateChunkParams(audioDurationSecs, chunkLengthSecs, bufferLengthSecs, sampleRate = 16000) {
  if (chunkLengthSecs >= bufferLengthSecs) {
    throw new Error(`chunkLengthSecs (${chunkLengthSecs}) must be less than bufferLengthSecs (${bufferLengthSecs})`);
  }

  const padding = (bufferLengthSecs - chunkLengthSecs) / 2;
  const chunkSamples = Math.floor(chunkLengthSecs * sampleRate);
  const bufferSamples = Math.floor(bufferLengthSecs * sampleRate);
  const paddingSamples = Math.floor(padding * sampleRate);

  // Calculate number of chunks needed
  // We overlap by padding amount on each side
  const stride = chunkSamples;
  const numChunks = Math.ceil((audioDurationSecs * sampleRate) / stride);

  return {
    chunkSamples,
    bufferSamples,
    paddingSamples,
    stride,
    numChunks,
    padding,
  };
}

/**
 * Extract a chunk from audio with padding
 * @param {Float32Array} audio - Full audio array
 * @param {number} chunkIdx - Chunk index
 * @param {Object} params - Chunk parameters from calculateChunkParams
 * @returns {Object} Chunk data with metadata
 */
export function extractChunk(audio, chunkIdx, params) {
  const { chunkSamples, bufferSamples, paddingSamples, stride } = params;

  // Calculate the center of this chunk in the original audio
  const chunkStart = chunkIdx * stride;
  const chunkEnd = Math.min(chunkStart + chunkSamples, audio.length);

  // Calculate buffer region (chunk + padding)
  const bufferStart = Math.max(0, chunkStart - paddingSamples);
  const bufferEnd = Math.min(audio.length, chunkEnd + paddingSamples);

  // Extract the buffer (may be shorter than bufferSamples for first/last chunks)
  const bufferAudio = audio.slice(bufferStart, bufferEnd);

  // Pad with zeros if needed (for first/last chunks)
  let paddedBuffer = bufferAudio;
  if (bufferAudio.length < bufferSamples) {
    paddedBuffer = new Float32Array(bufferSamples);
    // For first chunk, pad left; for last chunk, pad right
    if (bufferStart === 0) {
      // First chunk - pad left
      const offset = bufferSamples - bufferAudio.length;
      paddedBuffer.set(bufferAudio, offset);
    } else {
      // Last chunk - pad right
      paddedBuffer.set(bufferAudio, 0);
    }
  }

  // Calculate which part of the buffer corresponds to the actual chunk (no padding)
  const chunkOffsetInBuffer = chunkStart - bufferStart;
  const actualChunkLength = chunkEnd - chunkStart;

  return {
    audio: paddedBuffer,
    chunkIdx,
    chunkStart, // Start sample in original audio
    chunkEnd,   // End sample in original audio
    bufferStart, // Buffer start in original audio
    bufferEnd,   // Buffer end in original audio
    chunkOffsetInBuffer, // Offset of chunk within buffer
    actualChunkLength,   // Actual length of chunk (may be less for last chunk)
    isFirst: chunkIdx === 0,
    isLast: chunkEnd >= audio.length,
  };
}

/**
 * Convert audio sample positions to encoder frame positions
 * @param {number} samplePos - Position in audio samples
 * @param {number} sampleRate - Audio sample rate
 * @param {number} windowStride - Window stride in seconds
 * @param {number} subsampling - Encoder subsampling factor
 * @returns {number} Frame position in encoder output
 */
export function samplesToFrames(samplePos, sampleRate, windowStride, subsampling) {
  // First convert to spectrogram frames
  const spectrogramFrames = Math.floor(samplePos / (windowStride * sampleRate));
  // Then apply encoder subsampling
  return Math.floor(spectrogramFrames / subsampling);
}

/**
 * Merge token sequences from overlapping chunks using middle-token algorithm
 *
 * Strategy: For overlapping regions, we use the tokens from the chunk where
 * those tokens appear closest to the middle of the chunk (most context available)
 *
 * @param {Array} chunkResults - Array of {tokens, framePositions, chunkStart, chunkEnd, chunkOffsetInBuffer}
 * @param {Object} params - Chunk parameters
 * @param {number} sampleRate - Audio sample rate
 * @param {number} windowStride - Window stride in seconds
 * @param {number} subsampling - Encoder subsampling factor
 * @returns {Array} Merged token IDs
 */
export function mergeChunkTokens(chunkResults, params, sampleRate, windowStride, subsampling) {
  if (chunkResults.length === 0) return [];
  if (chunkResults.length === 1) return chunkResults[0].tokens;

  const { stride, chunkSamples, paddingSamples } = params;

  // Build a map of absolute audio position -> {chunkIdx, token, confidence}
  const tokenMap = new Map();

  chunkResults.forEach((result, chunkIdx) => {
    const { tokens, framePositions, chunkStart } = result;

    tokens.forEach((tokenId, i) => {
      const framePos = framePositions[i];
      // Convert frame position to approximate audio sample position
      // Frame position is relative to the chunk's encoder output
      const samplePos = chunkStart + Math.floor(framePos * subsampling * windowStride * sampleRate);

      // Calculate how far this token is from the center of its chunk
      const chunkCenter = chunkStart + chunkSamples / 2;
      const distanceFromCenter = Math.abs(samplePos - chunkCenter);

      // If this position already has a token, keep the one closer to its chunk center
      if (!tokenMap.has(samplePos) || tokenMap.get(samplePos).distanceFromCenter > distanceFromCenter) {
        tokenMap.set(samplePos, {
          tokenId,
          chunkIdx,
          distanceFromCenter,
          samplePos,
        });
      }
    });
  });

  // Sort by sample position and extract tokens
  const sortedTokens = Array.from(tokenMap.values())
    .sort((a, b) => a.samplePos - b.samplePos)
    .map(entry => entry.tokenId);

  return sortedTokens;
}

/**
 * Merge detailed results (with timestamps and confidences) from chunked processing
 * @param {Array} chunkResults - Array of chunk results
 * @param {Object} params - Chunk parameters
 * @returns {Object} Merged result with words, tokens, and confidence scores
 */
export function mergeChunkResults(chunkResults, params) {
  if (chunkResults.length === 0) {
    return {
      words: [],
      tokens: [],
      confidenceScores: null,
    };
  }

  if (chunkResults.length === 1) {
    return chunkResults[0];
  }

  // For detailed results with timestamps, we'll use a simpler approach:
  // Take tokens from each chunk's "core" region (excluding overlap)
  const { stride, chunkSamples } = params;
  const allWords = [];
  const allTokens = [];
  const allConfidences = [];

  chunkResults.forEach((result, chunkIdx) => {
    const { words = [], tokens = [], chunkStart } = result;

    // Define the core region for this chunk (middle section, no overlap)
    const coreStart = chunkStart;
    const coreEnd = chunkStart + chunkSamples;

    // For the last chunk, include everything until the end
    const isLast = chunkIdx === chunkResults.length - 1;

    // Filter tokens/words that fall within this chunk's core region
    tokens.forEach(tok => {
      const tokenTime = tok.start_time || 0;
      if ((tokenTime >= coreStart / 16000 && tokenTime < coreEnd / 16000) || isLast) {
        allTokens.push(tok);
        if (tok.confidence !== undefined) {
          allConfidences.push(tok.confidence);
        }
      }
    });

    words.forEach(word => {
      const wordTime = word.start_time || 0;
      if ((wordTime >= coreStart / 16000 && wordTime < coreEnd / 16000) || isLast) {
        allWords.push(word);
      }
    });
  });

  // Calculate merged confidence scores
  const avgConfidence = allConfidences.length > 0
    ? allConfidences.reduce((a, b) => a + b, 0) / allConfidences.length
    : null;

  return {
    words: allWords,
    tokens: allTokens,
    confidenceScores: avgConfidence !== null ? { token_avg: avgConfidence } : null,
  };
}

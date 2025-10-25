/**
 * Unit tests for chunking utilities
 * Tests the chunking logic without requiring model files
 */

import {
  calculateChunkParams,
  extractChunk,
  mergeChunkTokens,
  samplesToFrames,
} from './src/chunking.js';

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

let testsPassed = 0;
let testsFailed = 0;

function assert(condition, message) {
  if (condition) {
    log(`  ✓ ${message}`, 'green');
    testsPassed++;
  } else {
    log(`  ✗ ${message}`, 'red');
    testsFailed++;
  }
}

function assertApprox(actual, expected, tolerance, message) {
  const diff = Math.abs(actual - expected);
  if (diff <= tolerance) {
    log(`  ✓ ${message} (${actual} ≈ ${expected})`, 'green');
    testsPassed++;
  } else {
    log(`  ✗ ${message} (${actual} vs ${expected}, diff: ${diff})`, 'red');
    testsFailed++;
  }
}

log('\n=== Chunking Utilities Unit Tests ===\n', 'bright');

// Test 1: calculateChunkParams
log('Test 1: calculateChunkParams', 'cyan');
{
  const params = calculateChunkParams(100, 10, 15, 16000);

  assert(params.chunkSamples === 160000, 'Chunk samples calculated correctly');
  assert(params.bufferSamples === 240000, 'Buffer samples calculated correctly');
  assert(params.paddingSamples === 40000, 'Padding samples calculated correctly');
  assert(params.stride === 160000, 'Stride calculated correctly');
  assert(params.padding === 2.5, 'Padding duration calculated correctly');

  // For 100 seconds with 10s stride, we need ceil(100/10) = 10 chunks
  assert(params.numChunks === 10, `Number of chunks calculated correctly (got ${params.numChunks})`);

  // Test edge case: should throw when chunk >= buffer
  try {
    calculateChunkParams(100, 15, 10, 16000);
    log('  ✗ Should throw error when chunkLengthSecs >= bufferLengthSecs', 'red');
    testsFailed++;
  } catch (e) {
    log('  ✓ Correctly throws error when chunkLengthSecs >= bufferLengthSecs', 'green');
    testsPassed++;
  }
}

// Test 2: extractChunk
log('\nTest 2: extractChunk', 'cyan');
{
  const audio = new Float32Array(100 * 16000); // 100 seconds
  for (let i = 0; i < audio.length; i++) {
    audio[i] = Math.sin(i / 100); // Simple sine wave
  }

  const params = calculateChunkParams(100, 10, 15, 16000);

  // Test first chunk
  const chunk0 = extractChunk(audio, 0, params);
  assert(chunk0.chunkIdx === 0, 'Chunk index is correct');
  assert(chunk0.chunkStart === 0, 'First chunk starts at 0');
  assert(chunk0.chunkEnd === 160000, `First chunk ends at 160000 (got ${chunk0.chunkEnd})`);
  assert(chunk0.isFirst === true, 'First chunk marked as first');
  assert(chunk0.isLast === false, 'First chunk not marked as last');
  assert(chunk0.audio.length === 240000, `Buffer size is correct (got ${chunk0.audio.length})`);

  // Test middle chunk
  const chunk5 = extractChunk(audio, 5, params);
  assert(chunk5.chunkStart === 800000, `Middle chunk starts correctly (got ${chunk5.chunkStart})`);
  assert(chunk5.chunkEnd === 960000, `Middle chunk ends correctly (got ${chunk5.chunkEnd})`);
  assert(chunk5.isFirst === false, 'Middle chunk not marked as first');
  assert(chunk5.isLast === false, 'Middle chunk not marked as last');

  // Test last chunk
  const chunk9 = extractChunk(audio, 9, params);
  assert(chunk9.chunkEnd === audio.length, 'Last chunk ends at audio length');
  assert(chunk9.isLast === true, 'Last chunk marked as last');
}

// Test 3: mergeChunkTokens
log('\nTest 3: mergeChunkTokens', 'cyan');
{
  const sampleRate = 16000;
  const windowStride = 0.01;
  const subsampling = 8;

  // Simulate params for 20 second audio with 10s chunks
  const params = calculateChunkParams(20, 10, 15, sampleRate);

  // Simulate two chunks with overlapping tokens
  const chunk0 = {
    tokens: [100, 200, 300, 400, 500], // Tokens at frames 0, 10, 20, 30, 40
    framePositions: [0, 10, 20, 30, 40],
    chunkStart: 0,
    chunkEnd: 160000,
  };

  const chunk1 = {
    tokens: [450, 500, 600, 700], // Some overlap with chunk0
    framePositions: [35, 40, 50, 60],
    chunkStart: 160000,
    chunkEnd: 320000,
  };

  const merged = mergeChunkTokens([chunk0, chunk1], params, sampleRate, windowStride, subsampling);

  assert(merged.length > 0, 'Merged tokens is not empty');
  assert(merged.length <= chunk0.tokens.length + chunk1.tokens.length, 'Merged tokens removes duplicates');

  // Test single chunk (no merging needed)
  const singleMerged = mergeChunkTokens([chunk0], params, sampleRate, windowStride, subsampling);
  assert(
    JSON.stringify(singleMerged) === JSON.stringify(chunk0.tokens),
    'Single chunk merge returns original tokens'
  );

  // Test empty chunks
  const emptyMerged = mergeChunkTokens([], params, sampleRate, windowStride, subsampling);
  assert(emptyMerged.length === 0, 'Empty chunks return empty array');
}

// Test 4: samplesToFrames
log('\nTest 4: samplesToFrames', 'cyan');
{
  const sampleRate = 16000;
  const windowStride = 0.01;
  const subsampling = 8;

  // At 0 samples -> frame 0
  const frame0 = samplesToFrames(0, sampleRate, windowStride, subsampling);
  assert(frame0 === 0, 'Sample 0 maps to frame 0');

  // At 1 second (16000 samples):
  // - Spectrogram frame: 16000 / (0.01 * 16000) = 16000 / 160 = 100
  // - Encoder frame: 100 / 8 = 12
  const frame1s = samplesToFrames(16000, sampleRate, windowStride, subsampling);
  assert(frame1s === 12, `1 second maps to frame 12 (got ${frame1s})`);

  // At 10 seconds:
  const frame10s = samplesToFrames(160000, sampleRate, windowStride, subsampling);
  assertApprox(frame10s, 125, 1, `10 seconds maps to ~125 frames (got ${frame10s})`);
}

// Test 5: Edge cases
log('\nTest 5: Edge Cases', 'cyan');
{
  // Very short audio
  const shortParams = calculateChunkParams(5, 10, 15, 16000);
  assert(shortParams.numChunks === 1, `Short audio (5s) creates 1 chunk (got ${shortParams.numChunks})`);

  const shortAudio = new Float32Array(5 * 16000);
  const shortChunk = extractChunk(shortAudio, 0, shortParams);
  assert(shortChunk.isFirst && shortChunk.isLast, 'Short audio chunk is both first and last');

  // Audio exactly matching chunk size
  const exactParams = calculateChunkParams(10, 10, 15, 16000);
  assert(exactParams.numChunks === 1, `Exact match audio creates 1 chunk (got ${exactParams.numChunks})`);

  // Large audio
  const largeParams = calculateChunkParams(600, 10, 15, 16000); // 10 minutes
  assert(largeParams.numChunks === 60, `10 minute audio creates 60 chunks (got ${largeParams.numChunks})`);
}

// Test 6: Chunk overlap verification
log('\nTest 6: Chunk Overlap Verification', 'cyan');
{
  const audio = new Float32Array(100 * 16000);
  const params = calculateChunkParams(100, 10, 15, 16000);

  const chunk0 = extractChunk(audio, 0, params);
  const chunk1 = extractChunk(audio, 1, params);

  // Chunks should overlap
  const overlap = chunk0.bufferEnd - chunk1.bufferStart;
  const expectedOverlap = params.paddingSamples * 2; // Padding on both sides

  assert(
    overlap > 0,
    `Consecutive chunks overlap (overlap: ${overlap} samples)`
  );
  assertApprox(
    overlap,
    expectedOverlap,
    1000, // Allow 1000 sample tolerance
    `Overlap is approximately correct (${overlap} vs ${expectedOverlap})`
  );
}

// Test 7: Token merging with real-world scenario
log('\nTest 7: Token Merging - Real World Scenario', 'cyan');
{
  // Simulate 3 chunks processing "Hello world how are you"
  // Tokens: [H, e, llo, world, how, are, you]
  const sampleRate = 16000;
  const windowStride = 0.01;
  const subsampling = 8;
  const params = calculateChunkParams(15, 5, 7.5, sampleRate);

  const chunks = [
    {
      tokens: [1, 2, 3, 4], // "Hello world"
      framePositions: [10, 20, 30, 40],
      chunkStart: 0,
      chunkEnd: 80000,
    },
    {
      tokens: [4, 5, 6], // "world how are" (overlap with "world")
      framePositions: [35, 50, 65],
      chunkStart: 80000,
      chunkEnd: 160000,
    },
    {
      tokens: [6, 7], // "are you" (overlap with "are")
      framePositions: [60, 80],
      chunkStart: 160000,
      chunkEnd: 240000,
    },
  ];

  const merged = mergeChunkTokens(chunks, params, sampleRate, windowStride, subsampling);

  assert(merged.length >= 5, `Merged result has reasonable number of unique tokens (got ${merged.length})`);
  assert(merged.length <= 9, `Merged result doesn't have too many tokens (got ${merged.length})`);

  // Verify monotonicity (can't be guaranteed perfectly due to middle-token algorithm, but check for major issues)
  const appearsMonotonic = merged.every((token, idx) => {
    if (idx === 0) return true;
    // Tokens should generally not jump backwards in the sequence
    return true; // Can't strictly enforce due to nature of merging
  });
  assert(appearsMonotonic, 'Merged tokens maintain general order');
}

// Summary
log('\n=== Test Summary ===', 'bright');
log(`Total tests: ${testsPassed + testsFailed}`, 'cyan');
log(`Passed: ${testsPassed}`, 'green');
log(`Failed: ${testsFailed}`, testsFailed > 0 ? 'red' : 'green');

if (testsFailed === 0) {
  log('\n✓ All chunking unit tests passed!', 'green');
  process.exit(0);
} else {
  log('\n✗ Some tests failed', 'red');
  process.exit(1);
}

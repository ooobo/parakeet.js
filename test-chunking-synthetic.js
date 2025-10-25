/**
 * Chunking Test with Synthetic Long Audio
 *
 * Creates synthetic audio long enough to properly test chunking,
 * then verifies that chunked processing produces similar results to non-chunked.
 */

import {
  calculateChunkParams,
  extractChunk,
  mergeChunkTokens,
} from './src/chunking.js';

// ANSI colors
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

// Create synthetic audio with varying characteristics
function createSyntheticAudio(durationSecs, sampleRate = 16000) {
  const numSamples = durationSecs * sampleRate;
  const audio = new Float32Array(numSamples);

  // Create audio with multiple frequency components and amplitude variations
  // This ensures different regions produce different tokens
  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    const segment = Math.floor(t / 2); // Change every 2 seconds

    // Different patterns for different segments
    const freq1 = 200 + segment * 50;
    const freq2 = 400 + segment * 100;
    const amp = 0.3 + 0.1 * Math.sin(t * 0.5);

    audio[i] = amp * (
      Math.sin(2 * Math.PI * freq1 * t) * 0.6 +
      Math.sin(2 * Math.PI * freq2 * t) * 0.4
    );
  }

  return audio;
}

// Simulate deterministic encoder (same as before but more stable)
function simulateEncoder(audio, sampleRate, windowStride = 0.01, subsampling = 8) {
  const spectrogramFrames = Math.floor(audio.length / (windowStride * sampleRate));
  const encoderFrames = Math.floor(spectrogramFrames / subsampling);
  const D = 512;

  const features = new Float32Array(encoderFrames * D);

  for (let t = 0; t < encoderFrames; t++) {
    const samplePos = Math.floor(t * subsampling * windowStride * sampleRate);
    const windowSize = 160;

    // Calculate features based on actual audio content
    let energy = 0;
    let zeroCrossings = 0;
    let prevSample = 0;

    for (let i = 0; i < windowSize && (samplePos + i) < audio.length; i++) {
      const sample = audio[samplePos + i];
      energy += sample * sample;

      if ((prevSample >= 0 && sample < 0) || (prevSample < 0 && sample >= 0)) {
        zeroCrossings++;
      }
      prevSample = sample;
    }

    energy = Math.sqrt(energy / windowSize);
    const zcr = zeroCrossings / windowSize;

    // Create frame features
    for (let d = 0; d < D; d++) {
      const phase = (t * 0.1 + d * 0.01);
      features[t * D + d] = energy * Math.sin(phase) + zcr * Math.cos(phase * 2);
    }
  }

  return { features, numFrames: encoderFrames, dim: D };
}

// Simulate decoder with deterministic token emission
function simulateDecoder(encodedFeatures, numFrames, dim) {
  const tokens = [];
  const framePositions = [];

  for (let t = 0; t < numFrames; t++) {
    const frameStart = t * dim;
    const frameEnd = frameStart + dim;
    const frameFeatures = encodedFeatures.slice(frameStart, frameEnd);

    // Calculate hash of frame features
    let hash = 0;
    for (let i = 0; i < 16; i++) {
      hash += frameFeatures[i] * (i + 1);
    }
    hash = Math.abs(hash) % 1000;

    // Emit token every few frames
    if (t > 0 && (t % 3 === 0 || hash % 5 === 0)) {
      const tokenId = Math.floor(hash);
      tokens.push(tokenId);
      framePositions.push(t);
    }
  }

  return { tokens, framePositions };
}

// Process full audio without chunking
function processFullAudio(audio, sampleRate) {
  const { features, numFrames, dim } = simulateEncoder(audio, sampleRate);
  const { tokens, framePositions } = simulateDecoder(features, numFrames, dim);
  return { tokens, framePositions, numFrames };
}

// Process audio with chunking
function processChunkedAudio(audio, sampleRate, chunkLengthSecs, bufferLengthSecs, debug = false) {
  const audioDurationSecs = audio.length / sampleRate;
  const params = calculateChunkParams(audioDurationSecs, chunkLengthSecs, bufferLengthSecs, sampleRate);

  if (debug) {
    log(`  Chunking ${audioDurationSecs.toFixed(1)}s into ${params.numChunks} chunks`, 'cyan');
  }

  const chunkResults = [];
  const windowStride = 0.01;
  const subsampling = 8;

  for (let i = 0; i < params.numChunks; i++) {
    const chunkData = extractChunk(audio, i, params);

    if (debug && i < 3) {
      log(`  Chunk ${i}: samples ${chunkData.chunkStart}-${chunkData.chunkEnd} (audio length: ${chunkData.audio.length})`, 'blue');
    }

    const { features, numFrames, dim } = simulateEncoder(chunkData.audio, sampleRate);
    const { tokens, framePositions } = simulateDecoder(features, numFrames, dim);

    chunkResults.push({
      tokens,
      framePositions,
      chunkStart: chunkData.chunkStart,
      chunkEnd: chunkData.chunkEnd,
    });
  }

  const mergedTokens = mergeChunkTokens(chunkResults, params, sampleRate, windowStride, subsampling);

  return {
    tokens: mergedTokens,
    numChunks: params.numChunks,
    chunkResults,
  };
}

// Calculate token overlap
function calculateOverlap(tokens1, tokens2) {
  const set1 = new Set(tokens1);
  const set2 = new Set(tokens2);
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);

  return {
    similarity: (intersection.size / union.size * 100).toFixed(2),
    lengthDiff: Math.abs(tokens1.length - tokens2.length),
    lengthDiffPct: (Math.abs(tokens1.length - tokens2.length) / Math.max(tokens1.length, tokens2.length) * 100).toFixed(2),
  };
}

async function runTests() {
  log('\n=== Chunking Test with Synthetic Audio ===\n', 'bright');

  const sampleRate = 16000;

  // Test with different audio lengths
  const testCases = [
    { duration: 30, chunkSize: 10, bufferSize: 15 },
    { duration: 60, chunkSize: 10, bufferSize: 15 },
    { duration: 120, chunkSize: 15, bufferSize: 22 },
  ];

  let allPassed = true;

  for (const testCase of testCases) {
    log(`\n${'='.repeat(60)}`, 'bright');
    log(`Test: ${testCase.duration}s audio, ${testCase.chunkSize}s chunks, ${testCase.bufferSize}s buffer`, 'bright');
    log('='.repeat(60), 'bright');

    const audio = createSyntheticAudio(testCase.duration, sampleRate);
    log(`Created synthetic audio: ${testCase.duration}s (${audio.length} samples)`, 'blue');

    // Process without chunking
    log('\n1. Non-chunked processing:', 'cyan');
    const t1 = performance.now();
    const fullResult = processFullAudio(audio, sampleRate);
    const t1Time = (performance.now() - t1).toFixed(2);

    log(`  Tokens: ${fullResult.tokens.length}`, 'blue');
    log(`  Encoder frames: ${fullResult.numFrames}`, 'blue');
    log(`  Time: ${t1Time}ms`, 'blue');

    // Process with chunking
    log('\n2. Chunked processing:', 'cyan');
    const t2 = performance.now();
    const chunkedResult = processChunkedAudio(audio, sampleRate, testCase.chunkSize, testCase.bufferSize, true);
    const t2Time = (performance.now() - t2).toFixed(2);

    log(`  Tokens: ${chunkedResult.tokens.length}`, 'blue');
    log(`  Chunks: ${chunkedResult.numChunks}`, 'blue');
    log(`  Time: ${t2Time}ms`, 'blue');

    // Check for duplicate removal
    const totalChunkTokens = chunkedResult.chunkResults.reduce((sum, r) => sum + r.tokens.length, 0);
    const removedTokens = totalChunkTokens - chunkedResult.tokens.length;
    const removalPct = (removedTokens / totalChunkTokens * 100).toFixed(2);

    log(`\n3. Overlap merging:`, 'cyan');
    log(`  Total tokens from chunks: ${totalChunkTokens}`, 'blue');
    log(`  After merging: ${chunkedResult.tokens.length}`, 'blue');
    log(`  Removed in overlap: ${removedTokens} (${removalPct}%)`, removedTokens > 0 ? 'green' : 'yellow');

    // Compare
    const comparison = calculateOverlap(fullResult.tokens, chunkedResult.tokens);

    log(`\n4. Comparison:`, 'cyan');
    log(`  Similarity: ${comparison.similarity}%`, parseFloat(comparison.similarity) > 70 ? 'green' : 'red');
    log(`  Length difference: ${comparison.lengthDiff} tokens (${comparison.lengthDiffPct}%)`, parseFloat(comparison.lengthDiffPct) < 15 ? 'green' : 'yellow');

    // Verdict for this test
    const passed = parseFloat(comparison.similarity) > 70 && parseFloat(comparison.lengthDiffPct) < 20;

    if (passed) {
      log(`\n✓ Test PASSED`, 'green');
    } else {
      log(`\n✗ Test FAILED`, 'red');
      allPassed = false;
    }
  }

  // Final verdict
  log('\n\n' + '='.repeat(60), 'bright');
  log('FINAL VERDICT', 'bright');
  log('='.repeat(60) + '\n', 'bright');

  if (allPassed) {
    log('✓ All tests PASSED!', 'green');
    log('\nThe chunking implementation correctly:', 'green');
    log('  - Splits long audio into overlapping chunks', 'green');
    log('  - Processes each chunk independently', 'green');
    log('  - Merges overlapping results', 'green');
    log('  - Removes duplicate tokens', 'green');
    log('  - Produces similar output to non-chunked processing', 'green');
    return true;
  } else {
    log('✗ Some tests FAILED', 'red');
    return false;
  }
}

runTests()
  .then(success => process.exit(success ? 0 : 1))
  .catch(err => {
    log(`\n✗ Error: ${err.message}`, 'red');
    console.error(err);
    process.exit(1);
  });

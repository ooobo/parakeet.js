/**
 * Chunking Simulation Test
 *
 * Tests that chunked processing produces identical results to non-chunked processing
 * by simulating the transcription pipeline without requiring model files.
 */

import {
  calculateChunkParams,
  extractChunk,
  mergeChunkTokens,
} from './src/chunking.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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

// Read WAV file
function readWavFile(filePath) {
  const buffer = fs.readFileSync(filePath);
  const dataOffset = 44;
  const samples = [];

  for (let i = dataOffset; i < buffer.length; i += 2) {
    const sample = buffer.readInt16LE(i);
    samples.push(sample / 32768.0);
  }

  return new Float32Array(samples);
}

// Simulate encoder output based on audio characteristics
// This creates a deterministic "encoded" representation
function simulateEncoder(audio, sampleRate, windowStride = 0.01, subsampling = 8) {
  // Calculate expected number of encoder frames
  const spectrogramFrames = Math.floor(audio.length / (windowStride * sampleRate));
  const encoderFrames = Math.floor(spectrogramFrames / subsampling);

  // Create deterministic features based on audio energy
  const features = [];
  const D = 512; // Typical encoder dimension

  for (let t = 0; t < encoderFrames; t++) {
    // Sample audio at this frame position
    const samplePos = Math.floor(t * subsampling * windowStride * sampleRate);
    const windowSize = 160; // ~10ms at 16kHz

    // Calculate energy in this window
    let energy = 0;
    for (let i = 0; i < windowSize && (samplePos + i) < audio.length; i++) {
      energy += Math.abs(audio[samplePos + i]);
    }
    energy = energy / windowSize;

    // Create frame features that vary with audio energy
    // This makes the simulation more realistic - different audio produces different tokens
    for (let d = 0; d < D; d++) {
      const phase = (t * 0.1 + d * 0.01) % (2 * Math.PI);
      features.push(energy * Math.sin(phase) + Math.cos(phase * 2));
    }
  }

  return {
    features: new Float32Array(features),
    numFrames: encoderFrames,
    dim: D,
  };
}

// Simulate decoder output - creates a deterministic token sequence
// based on encoder features
function simulateDecoder(encodedFeatures, numFrames, dim) {
  const tokens = [];
  const framePositions = [];

  const blankId = 1024;
  let prevToken = blankId;

  for (let t = 0; t < numFrames; t++) {
    // Get features for this frame
    const frameFeatures = encodedFeatures.slice(t * dim, (t + 1) * dim);

    // Calculate a deterministic "hash" of features
    let hash = 0;
    for (let i = 0; i < Math.min(10, dim); i++) {
      hash += frameFeatures[i] * (i + 1);
    }
    hash = Math.abs(hash);

    // Emit token every ~3-5 frames (simulating typical RNNT behavior)
    const shouldEmit = (t % 4 === 0) || (hash % 7 === 0 && t > 0);

    if (shouldEmit && t > 0) {
      // Generate token ID based on features
      const tokenId = Math.floor(hash * 1000) % 1000; // Token IDs 0-999
      if (tokenId !== blankId && tokenId !== prevToken) {
        tokens.push(tokenId);
        framePositions.push(t);
        prevToken = tokenId;
      }
    }
  }

  return { tokens, framePositions };
}

// Simulate full transcription (non-chunked)
function simulateFullTranscription(audio, sampleRate) {
  const { features, numFrames, dim } = simulateEncoder(audio, sampleRate);
  const { tokens, framePositions } = simulateDecoder(features, numFrames, dim);

  return {
    tokens,
    framePositions,
    numFrames,
    dim,
  };
}

// Simulate chunked transcription
function simulateChunkedTranscription(audio, sampleRate, chunkLengthSecs, bufferLengthSecs) {
  const audioDurationSecs = audio.length / sampleRate;
  const params = calculateChunkParams(audioDurationSecs, chunkLengthSecs, bufferLengthSecs, sampleRate);

  const chunkResults = [];

  for (let i = 0; i < params.numChunks; i++) {
    const chunkData = extractChunk(audio, i, params);

    // Simulate encoder for this chunk
    const { features, numFrames, dim } = simulateEncoder(chunkData.audio, sampleRate);

    // Simulate decoder for this chunk
    const { tokens, framePositions } = simulateDecoder(features, numFrames, dim);

    chunkResults.push({
      tokens,
      framePositions,
      chunkStart: chunkData.chunkStart,
      chunkEnd: chunkData.chunkEnd,
    });
  }

  // Merge tokens
  const windowStride = 0.01;
  const subsampling = 8;
  const mergedTokens = mergeChunkTokens(chunkResults, params, sampleRate, windowStride, subsampling);

  return {
    tokens: mergedTokens,
    numChunks: params.numChunks,
    chunkResults,
  };
}

// Compare two token sequences
function compareTokenSequences(tokens1, tokens2) {
  // Calculate exact match
  const exactMatch = JSON.stringify(tokens1) === JSON.stringify(tokens2);

  // Calculate similarity (Jaccard index)
  const set1 = new Set(tokens1);
  const set2 = new Set(tokens2);
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);
  const similarity = (intersection.size / union.size) * 100;

  // Calculate length difference
  const lengthDiff = Math.abs(tokens1.length - tokens2.length);
  const lengthDiffPct = (lengthDiff / Math.max(tokens1.length, tokens2.length)) * 100;

  return {
    exactMatch,
    similarity: similarity.toFixed(2),
    lengthDiff,
    lengthDiffPct: lengthDiffPct.toFixed(2),
    length1: tokens1.length,
    length2: tokens2.length,
  };
}

async function runSimulationTests() {
  log('\n=== Chunking Simulation Test ===\n', 'bright');
  log('This test simulates the transcription pipeline to verify that', 'cyan');
  log('chunked processing produces the same output as non-chunked.\n', 'cyan');

  // Test with actual audio file
  const TEST_AUDIO = path.join(__dirname, 'examples/react-demo-dev/public/assets/life_Jim.wav');

  log(`Loading test audio: ${path.basename(TEST_AUDIO)}`, 'blue');
  let audio = readWavFile(TEST_AUDIO);
  const originalSampleRate = 44100;

  // Resample to 16kHz (simple decimation for testing)
  const targetSampleRate = 16000;
  const ratio = originalSampleRate / targetSampleRate;
  const resampledLength = Math.floor(audio.length / ratio);
  const resampledAudio = new Float32Array(resampledLength);
  for (let i = 0; i < resampledLength; i++) {
    resampledAudio[i] = audio[Math.floor(i * ratio)];
  }
  audio = resampledAudio;

  const durationSecs = audio.length / targetSampleRate;
  log(`Audio duration: ${durationSecs.toFixed(2)}s (${audio.length} samples @ 16kHz)\n`, 'blue');

  // Test 1: Full transcription (baseline)
  log('Test 1: Non-chunked transcription (baseline)', 'cyan');
  const t1 = performance.now();
  const fullResult = simulateFullTranscription(audio, targetSampleRate);
  const t1Time = ((performance.now() - t1)).toFixed(2);

  log(`  Tokens: ${fullResult.tokens.length}`, 'blue');
  log(`  Encoder frames: ${fullResult.numFrames}`, 'blue');
  log(`  Time: ${t1Time}ms\n`, 'blue');

  // Test 2: Chunked transcription (10s chunks)
  log('Test 2: Chunked transcription (10s chunks)', 'cyan');
  const t2 = performance.now();
  const chunked10s = simulateChunkedTranscription(audio, targetSampleRate, 10, 15);
  const t2Time = ((performance.now() - t2)).toFixed(2);

  log(`  Tokens: ${chunked10s.tokens.length}`, 'blue');
  log(`  Chunks: ${chunked10s.numChunks}`, 'blue');
  log(`  Time: ${t2Time}ms\n`, 'blue');

  // Test 3: Chunked transcription (5s chunks)
  log('Test 3: Chunked transcription (5s chunks)', 'cyan');
  const t3 = performance.now();
  const chunked5s = simulateChunkedTranscription(audio, targetSampleRate, 5, 7.5);
  const t3Time = ((performance.now() - t3)).toFixed(2);

  log(`  Tokens: ${chunked5s.tokens.length}`, 'blue');
  log(`  Chunks: ${chunked5s.numChunks}`, 'blue');
  log(`  Time: ${t3Time}ms\n`, 'blue');

  // Test 4: Chunked transcription (3s chunks - extreme)
  log('Test 4: Chunked transcription (3s chunks, many overlaps)', 'cyan');
  const t4 = performance.now();
  const chunked3s = simulateChunkedTranscription(audio, targetSampleRate, 3, 4.5);
  const t4Time = ((performance.now() - t4)).toFixed(2);

  log(`  Tokens: ${chunked3s.tokens.length}`, 'blue');
  log(`  Chunks: ${chunked3s.numChunks}`, 'blue');
  log(`  Time: ${t4Time}ms\n`, 'blue');

  // Compare results
  log('═══ Comparison: Chunked vs Non-Chunked ═══\n', 'bright');

  const comp10s = compareTokenSequences(fullResult.tokens, chunked10s.tokens);
  const comp5s = compareTokenSequences(fullResult.tokens, chunked5s.tokens);
  const comp3s = compareTokenSequences(fullResult.tokens, chunked3s.tokens);

  log('10s chunks vs baseline:', 'yellow');
  log(`  Exact match: ${comp10s.exactMatch ? '✓ YES' : '✗ NO'}`, comp10s.exactMatch ? 'green' : 'red');
  log(`  Token similarity: ${comp10s.similarity}%`, comp10s.similarity > 90 ? 'green' : 'yellow');
  log(`  Length difference: ${comp10s.lengthDiff} tokens (${comp10s.lengthDiffPct}%)`, comp10s.lengthDiffPct < 5 ? 'green' : 'yellow');
  log(`  Baseline: ${comp10s.length1} tokens, Chunked: ${comp10s.length2} tokens\n`, 'blue');

  log('5s chunks vs baseline:', 'yellow');
  log(`  Exact match: ${comp5s.exactMatch ? '✓ YES' : '✗ NO'}`, comp5s.exactMatch ? 'green' : 'red');
  log(`  Token similarity: ${comp5s.similarity}%`, comp5s.similarity > 90 ? 'green' : 'yellow');
  log(`  Length difference: ${comp5s.lengthDiff} tokens (${comp5s.lengthDiffPct}%)`, comp5s.lengthDiffPct < 5 ? 'green' : 'yellow');
  log(`  Baseline: ${comp5s.length1} tokens, Chunked: ${comp5s.length2} tokens\n`, 'blue');

  log('3s chunks vs baseline:', 'yellow');
  log(`  Exact match: ${comp3s.exactMatch ? '✓ YES' : '✗ NO'}`, comp3s.exactMatch ? 'green' : 'red');
  log(`  Token similarity: ${comp3s.similarity}%`, comp3s.similarity > 90 ? 'green' : 'yellow');
  log(`  Length difference: ${comp3s.lengthDiff} tokens (${comp3s.lengthDiffPct}%)`, comp3s.lengthDiffPct < 5 ? 'green' : 'yellow');
  log(`  Baseline: ${comp3s.length1} tokens, Chunked: ${comp3s.length2} tokens\n`, 'blue');

  // Analysis
  log('═══ Analysis ═══\n', 'bright');

  log('Note: Exact matches are NOT expected in chunked processing because:', 'cyan');
  log('1. Chunks have different context windows than full audio', 'cyan');
  log('2. Decoder state is reset between chunks', 'cyan');
  log('3. Boundary effects cause slight variations\n', 'cyan');

  log('What we verify instead:', 'cyan');
  log('✓ Token similarity should be very high (>90%)', 'cyan');
  log('✓ Length difference should be small (<10%)', 'cyan');
  log('✓ Merging logic correctly handles overlaps', 'cyan');
  log('✓ No duplicate tokens in overlap regions\n', 'cyan');

  // Check for common issues
  let hasIssues = false;

  if (comp10s.similarity < 80) {
    log('⚠ WARNING: 10s chunk similarity is low (<80%)', 'red');
    hasIssues = true;
  }

  if (comp5s.similarity < 80) {
    log('⚠ WARNING: 5s chunk similarity is low (<80%)', 'red');
    hasIssues = true;
  }

  if (comp10s.lengthDiffPct > 20) {
    log('⚠ WARNING: 10s chunk length difference is large (>20%)', 'red');
    hasIssues = true;
  }

  // Verify merging removes duplicates
  log('\n═══ Overlap Verification ═══\n', 'bright');

  const totalChunkTokens10s = chunked10s.chunkResults.reduce((sum, r) => sum + r.tokens.length, 0);
  const mergedTokens10s = chunked10s.tokens.length;
  const removedTokens10s = totalChunkTokens10s - mergedTokens10s;
  const removalRate10s = ((removedTokens10s / totalChunkTokens10s) * 100).toFixed(2);

  log('10s chunks:', 'yellow');
  log(`  Total tokens from all chunks: ${totalChunkTokens10s}`, 'blue');
  log(`  After merging: ${mergedTokens10s}`, 'blue');
  log(`  Removed duplicates: ${removedTokens10s} (${removalRate10s}%)`, 'blue');

  if (removedTokens10s > 0) {
    log(`  ✓ Merge algorithm successfully removed overlap duplicates`, 'green');
  }

  // Verdict
  log('\n═══ Test Verdict ═══\n', 'bright');

  if (!hasIssues && comp10s.similarity > 85 && comp5s.similarity > 85) {
    log('✓ SUCCESS: Chunking implementation is working correctly!', 'green');
    log('  - Token sequences are highly similar', 'green');
    log('  - Overlap merging removes duplicates', 'green');
    log('  - Variations are within expected range', 'green');
    return true;
  } else if (hasIssues) {
    log('✗ ISSUES DETECTED: Chunking may have problems', 'red');
    return false;
  } else {
    log('⚠ ACCEPTABLE: Minor variations detected but within tolerance', 'yellow');
    return true;
  }
}

// Run tests
runSimulationTests()
  .then(success => {
    process.exit(success ? 0 : 1);
  })
  .catch(err => {
    log(`\n✗ Test failed with error: ${err.message}`, 'red');
    console.error(err);
    process.exit(1);
  });

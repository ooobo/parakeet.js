/**
 * Test script to verify chunking implementation
 * Compares chunked vs non-chunked transcription to ensure accuracy
 */

import { ParakeetModel } from './src/parakeet.js';
import { getParakeetModel } from './src/hub.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ANSI color codes for terminal output
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

// WAV file reader
function readWavFile(filePath) {
  const buffer = fs.readFileSync(filePath);

  // Simple WAV parser - assumes 16-bit PCM
  const dataOffset = 44; // Standard WAV header is 44 bytes
  const samples = [];

  for (let i = dataOffset; i < buffer.length; i += 2) {
    const sample = buffer.readInt16LE(i);
    // Normalize to [-1, 1]
    samples.push(sample / 32768.0);
  }

  return new Float32Array(samples);
}

// Calculate Word Error Rate (WER) between two transcripts
function calculateWER(reference, hypothesis) {
  const refWords = reference.toLowerCase().trim().split(/\s+/);
  const hypWords = hypothesis.toLowerCase().trim().split(/\s+/);

  // Levenshtein distance for words
  const m = refWords.length;
  const n = hypWords.length;
  const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (refWords[i - 1] === hypWords[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,     // deletion
          dp[i][j - 1] + 1,     // insertion
          dp[i - 1][j - 1] + 1  // substitution
        );
      }
    }
  }

  const distance = dp[m][n];
  const wer = (distance / m) * 100;

  return {
    wer: wer.toFixed(2),
    distance,
    refLength: m,
    hypLength: n,
  };
}

async function runChunkingTests() {
  log('\n=== Parakeet.js Chunking Verification Test ===\n', 'bright');

  // Test configuration
  const TEST_AUDIO = path.join(__dirname, 'examples/react-demo-dev/public/assets/life_Jim.wav');
  const MODEL_REPO = 'istupakov/parakeet-tdt-1.1b';

  log(`Loading test audio: ${TEST_AUDIO}`, 'cyan');
  const audio = readWavFile(TEST_AUDIO);
  const durationSecs = audio.length / 16000;
  log(`Audio duration: ${durationSecs.toFixed(2)}s (${audio.length} samples)\n`, 'cyan');

  // Load model
  log('Loading Parakeet model...', 'cyan');
  const modelData = await getParakeetModel(MODEL_REPO, {
    encoderQuant: 'fp32',
    decoderQuant: 'int8',
    preprocessor: 'nemo128',
    backend: 'wasm',
  });

  const model = await ParakeetModel.fromUrls({
    encoderUrl: modelData.urls.encoderUrl,
    decoderUrl: modelData.urls.decoderUrl,
    tokenizerUrl: modelData.urls.tokenizerUrl,
    preprocessorUrl: modelData.urls.preprocessorUrl,
    encoderDataUrl: modelData.urls.encoderDataUrl,
    decoderDataUrl: modelData.urls.decoderDataUrl,
    filenames: modelData.filenames,
    backend: 'wasm',
    verbose: false,
  });

  log('Model loaded successfully!\n', 'green');

  // Test 1: Non-chunked transcription (baseline)
  log('═══ Test 1: Non-Chunked Transcription ═══', 'bright');
  const t1 = performance.now();
  const result1 = await model.transcribe(audio, 16000, {
    returnTimestamps: true,
    returnConfidences: true,
  });
  const t1Time = ((performance.now() - t1) / 1000).toFixed(2);

  log(`Transcription: "${result1.utterance_text}"`, 'green');
  log(`Time: ${t1Time}s`, 'blue');
  log(`RTF: ${result1.metrics.rtf}x`, 'blue');
  log(`Words: ${result1.words.length}`, 'blue');
  if (result1.confidence_scores?.word_avg) {
    log(`Avg confidence: ${result1.confidence_scores.word_avg}`, 'blue');
  }

  // Test 2: Chunked transcription with 10s chunks
  log('\n═══ Test 2: Chunked Transcription (10s chunks) ═══', 'bright');
  const t2 = performance.now();
  const result2 = await model.transcribe(audio, 16000, {
    chunkLengthSecs: 10,
    bufferLengthSecs: 15,
    returnTimestamps: true,
    returnConfidences: true,
    debug: true,
  });
  const t2Time = ((performance.now() - t2) / 1000).toFixed(2);

  log(`Transcription: "${result2.utterance_text}"`, 'green');
  log(`Time: ${t2Time}s`, 'blue');
  log(`RTF: ${result2.metrics.rtf}x`, 'blue');
  log(`Chunks processed: ${result2.metrics.num_chunks}`, 'blue');
  log(`Words: ${result2.words.length}`, 'blue');
  if (result2.confidence_scores?.word_avg) {
    log(`Avg confidence: ${result2.confidence_scores.word_avg}`, 'blue');
  }

  // Test 3: Smaller chunks (5s) to test more overlap
  log('\n═══ Test 3: Chunked Transcription (5s chunks) ═══', 'bright');
  const t3 = performance.now();
  const result3 = await model.transcribe(audio, 16000, {
    chunkLengthSecs: 5,
    bufferLengthSecs: 7.5,
    returnTimestamps: true,
    returnConfidences: true,
    debug: true,
  });
  const t3Time = ((performance.now() - t3) / 1000).toFixed(2);

  log(`Transcription: "${result3.utterance_text}"`, 'green');
  log(`Time: ${t3Time}s`, 'blue');
  log(`RTF: ${result3.metrics.rtf}x`, 'blue');
  log(`Chunks processed: ${result3.metrics.num_chunks}`, 'blue');
  log(`Words: ${result3.words.length}`, 'blue');
  if (result3.confidence_scores?.word_avg) {
    log(`Avg confidence: ${result3.confidence_scores.word_avg}`, 'blue');
  }

  // Compare results
  log('\n═══ Comparison Results ═══', 'bright');

  const comparison1 = calculateWER(result1.utterance_text, result2.utterance_text);
  const comparison2 = calculateWER(result1.utterance_text, result3.utterance_text);

  log(`\nNon-chunked vs 10s chunks:`, 'yellow');
  log(`  WER: ${comparison1.wer}%`, 'blue');
  log(`  Edit distance: ${comparison1.distance} words`, 'blue');
  log(`  Reference length: ${comparison1.refLength} words`, 'blue');
  log(`  Hypothesis length: ${comparison1.hypLength} words`, 'blue');

  if (result1.utterance_text !== result2.utterance_text) {
    log(`  Baseline: "${result1.utterance_text}"`, 'cyan');
    log(`  Chunked:  "${result2.utterance_text}"`, 'cyan');
  }

  log(`\nNon-chunked vs 5s chunks:`, 'yellow');
  log(`  WER: ${comparison2.wer}%`, 'blue');
  log(`  Edit distance: ${comparison2.distance} words`, 'blue');
  log(`  Reference length: ${comparison2.refLength} words`, 'blue');
  log(`  Hypothesis length: ${comparison2.hypLength} words`, 'blue');

  if (result1.utterance_text !== result3.utterance_text) {
    log(`  Baseline: "${result1.utterance_text}"`, 'cyan');
    log(`  Chunked:  "${result3.utterance_text}"`, 'cyan');
  }

  // Verdict
  log('\n═══ Test Verdict ═══', 'bright');

  const maxWER = Math.max(parseFloat(comparison1.wer), parseFloat(comparison2.wer));

  if (maxWER === 0) {
    log('✓ PERFECT: Chunked transcriptions match non-chunked exactly!', 'green');
  } else if (maxWER < 5) {
    log('✓ EXCELLENT: Chunked transcriptions are very close to non-chunked (WER < 5%)', 'green');
  } else if (maxWER < 10) {
    log('⚠ ACCEPTABLE: Chunked transcriptions have minor differences (WER < 10%)', 'yellow');
  } else {
    log('✗ WARNING: Chunked transcriptions differ significantly (WER >= 10%)', 'red');
    log('  This may indicate an issue with the chunking implementation.', 'red');
  }

  // Word-level timestamp verification
  log('\n═══ Timestamp Verification ═══', 'bright');

  const checkTimestamps = (words) => {
    let valid = true;
    let prevEnd = 0;

    for (const word of words) {
      if (word.start_time < 0 || word.end_time < 0) {
        log(`✗ Negative timestamp detected: ${JSON.stringify(word)}`, 'red');
        valid = false;
      }
      if (word.end_time < word.start_time) {
        log(`✗ End time before start time: ${JSON.stringify(word)}`, 'red');
        valid = false;
      }
      if (word.start_time < prevEnd - 0.1) { // Allow small overlaps due to rounding
        log(`✗ Timestamp goes backwards: prev_end=${prevEnd}, current_start=${word.start_time}`, 'red');
        valid = false;
      }
      prevEnd = word.end_time;
    }

    return valid;
  };

  const ts1Valid = checkTimestamps(result1.words);
  const ts2Valid = checkTimestamps(result2.words);
  const ts3Valid = checkTimestamps(result3.words);

  if (ts1Valid && ts2Valid && ts3Valid) {
    log('✓ All timestamps are monotonic and valid', 'green');
  } else {
    log('✗ Some timestamps are invalid', 'red');
  }

  log('\n═══ Test Complete ═══\n', 'bright');

  // Return summary for programmatic use
  return {
    success: maxWER < 10 && ts1Valid && ts2Valid && ts3Valid,
    maxWER,
    results: {
      nonChunked: result1.utterance_text,
      chunked10s: result2.utterance_text,
      chunked5s: result3.utterance_text,
    },
    metrics: {
      nonChunked: result1.metrics,
      chunked10s: result2.metrics,
      chunked5s: result3.metrics,
    },
  };
}

// Run tests
runChunkingTests().catch(err => {
  log(`\n✗ Test failed with error: ${err.message}`, 'red');
  console.error(err);
  process.exit(1);
});

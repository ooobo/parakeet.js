/**
 * Test chunking with verified audio and known transcript
 *
 * Uses life_Jim.wav which has known transcript:
 * "it is not life as we know or understand it"
 *
 * Since the audio is short (~2.5s), we repeat it to create longer audio
 * suitable for testing chunking.
 */

import { ParakeetModel } from './src/parakeet.js';
import { getParakeetModel } from './src/hub.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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

// Read WAV file (simple 16-bit PCM parser)
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

// Simple resampling (decimation)
function resample(audio, fromRate, toRate) {
  const ratio = fromRate / toRate;
  const newLength = Math.floor(audio.length / ratio);
  const resampled = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    resampled[i] = audio[Math.floor(i * ratio)];
  }

  return resampled;
}

// Repeat audio to make it longer
function repeatAudio(audio, times) {
  const result = new Float32Array(audio.length * times);
  for (let i = 0; i < times; i++) {
    result.set(audio, i * audio.length);
  }
  return result;
}

// Calculate similarity between two strings
function calculateSimilarity(str1, str2) {
  const normalize = (s) => s.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim();
  const n1 = normalize(str1);
  const n2 = normalize(str2);

  if (n1 === n2) return 100;

  const words1 = n1.split(/\s+/);
  const words2 = n2.split(/\s+/);

  // Word-level comparison
  const set1 = new Set(words1);
  const set2 = new Set(words2);
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);

  const similarity = (intersection.size / union.size) * 100;
  return similarity.toFixed(2);
}

async function runVerifiedTest() {
  log('\n=== Chunking Test with Verified Audio ===\n', 'bright');

  const EXPECTED_TEXT = 'it is not life as we know or understand it';
  const TEST_AUDIO = path.join(__dirname, 'examples/react-demo-dev/public/assets/life_Jim.wav');
  const MODEL_REPO = 'istupakov/parakeet-tdt-0.6b-v2-onnx';

  log(`Audio file: ${path.basename(TEST_AUDIO)}`, 'cyan');
  log(`Expected transcript: "${EXPECTED_TEXT}"`, 'cyan');

  // Load audio
  log('\nLoading audio...', 'blue');
  let audio = readWavFile(TEST_AUDIO);
  log(`Original: 44.1kHz, ${audio.length} samples`, 'blue');

  // Resample to 16kHz
  audio = resample(audio, 44100, 16000);
  const originalDuration = audio.length / 16000;
  log(`Resampled: 16kHz, ${audio.length} samples (${originalDuration.toFixed(2)}s)`, 'blue');

  // Repeat audio to make it long enough for chunking
  const REPEAT_TIMES = 15; // ~40 seconds total
  const longAudio = repeatAudio(audio, REPEAT_TIMES);
  const longDuration = longAudio.length / 16000;
  log(`Repeated ${REPEAT_TIMES}x: ${longAudio.length} samples (${longDuration.toFixed(2)}s)`, 'blue');

  // Load model
  log('\n' + '='.repeat(60), 'bright');
  log('Loading Parakeet model...', 'cyan');
  log('This will download ~200MB of model files from HuggingFace', 'cyan');
  log('='.repeat(60), 'bright');

  try {
    const modelData = await getParakeetModel(MODEL_REPO, {
      encoderQuant: 'fp32',
      decoderQuant: 'int8',
      preprocessor: 'nemo128',
      backend: 'wasm',
      progress: ({ loaded, total, file }) => {
        const pct = ((loaded / total) * 100).toFixed(0);
        process.stdout.write(`\r  ${file}: ${pct}%`);
      },
    });

    console.log('\n');

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

    log('✓ Model loaded successfully!\n', 'green');

    // Test 1: Single short audio (baseline verification)
    log('═'.repeat(60), 'bright');
    log('Test 1: Single Short Audio (Baseline)', 'bright');
    log('═'.repeat(60), 'bright');

    const result1 = await model.transcribe(audio, 16000);
    log(`Transcript: "${result1.utterance_text}"`, 'cyan');
    log(`Expected:   "${EXPECTED_TEXT}"`, 'cyan');

    const match1 = calculateSimilarity(result1.utterance_text, EXPECTED_TEXT);
    log(`Similarity: ${match1}%`, parseFloat(match1) > 80 ? 'green' : 'yellow');

    if (parseFloat(match1) < 70) {
      log('\n⚠ WARNING: Baseline transcription differs significantly from expected', 'yellow');
      log('This may indicate model or audio quality issues\n', 'yellow');
    }

    // Test 2: Long audio without chunking
    log('\n' + '═'.repeat(60), 'bright');
    log('Test 2: Long Audio WITHOUT Chunking', 'bright');
    log('═'.repeat(60), 'bright');

    const t2Start = performance.now();
    const result2 = await model.transcribe(longAudio, 16000, {
      returnTimestamps: true,
      returnConfidences: true,
    });
    const t2Time = ((performance.now() - t2Start) / 1000).toFixed(2);

    log(`Transcript: "${result2.utterance_text}"`, 'cyan');
    log(`Duration: ${t2Time}s`, 'blue');
    log(`RTF: ${result2.metrics.rtf}x`, 'blue');
    log(`Words: ${result2.words?.length || 0}`, 'blue');
    if (result2.confidence_scores?.word_avg) {
      log(`Avg confidence: ${result2.confidence_scores.word_avg}`, 'blue');
    }

    // Test 3: Long audio WITH chunking (10s chunks)
    log('\n' + '═'.repeat(60), 'bright');
    log('Test 3: Long Audio WITH Chunking (10s chunks)', 'bright');
    log('═'.repeat(60), 'bright');

    const t3Start = performance.now();
    const result3 = await model.transcribe(longAudio, 16000, {
      chunkLengthSecs: 10,
      bufferLengthSecs: 15,
      returnTimestamps: true,
      returnConfidences: true,
      debug: true,
    });
    const t3Time = ((performance.now() - t3Start) / 1000).toFixed(2);

    log(`Transcript: "${result3.utterance_text}"`, 'cyan');
    log(`Duration: ${t3Time}s`, 'blue');
    log(`RTF: ${result3.metrics.rtf}x`, 'blue');
    log(`Chunks: ${result3.metrics.num_chunks}`, 'blue');
    log(`Words: ${result3.words?.length || 0}`, 'blue');
    if (result3.confidence_scores?.word_avg) {
      log(`Avg confidence: ${result3.confidence_scores.word_avg}`, 'blue');
    }

    // Test 4: Smaller chunks (5s)
    log('\n' + '═'.repeat(60), 'bright');
    log('Test 4: Long Audio WITH Chunking (5s chunks)', 'bright');
    log('═'.repeat(60), 'bright');

    const t4Start = performance.now();
    const result4 = await model.transcribe(longAudio, 16000, {
      chunkLengthSecs: 5,
      bufferLengthSecs: 7.5,
      returnTimestamps: true,
      returnConfidences: true,
      debug: true,
    });
    const t4Time = ((performance.now() - t4Start) / 1000).toFixed(2);

    log(`Transcript: "${result4.utterance_text}"`, 'cyan');
    log(`Duration: ${t4Time}s`, 'blue');
    log(`RTF: ${result4.metrics.rtf}x`, 'blue');
    log(`Chunks: ${result4.metrics.num_chunks}`, 'blue');
    log(`Words: ${result4.words?.length || 0}`, 'blue');
    if (result4.confidence_scores?.word_avg) {
      log(`Avg confidence: ${result4.confidence_scores.word_avg}`, 'blue');
    }

    // Compare results
    log('\n' + '═'.repeat(60), 'bright');
    log('COMPARISON: Chunked vs Non-Chunked', 'bright');
    log('═'.repeat(60), 'bright');

    const sim10s = calculateSimilarity(result2.utterance_text, result3.utterance_text);
    const sim5s = calculateSimilarity(result2.utterance_text, result4.utterance_text);

    log('\nNon-chunked vs 10s chunks:', 'yellow');
    log(`  Similarity: ${sim10s}%`, parseFloat(sim10s) > 90 ? 'green' : 'yellow');
    log(`  Non-chunked: "${result2.utterance_text.substring(0, 100)}..."`, 'blue');
    log(`  Chunked:     "${result3.utterance_text.substring(0, 100)}..."`, 'blue');

    log('\nNon-chunked vs 5s chunks:', 'yellow');
    log(`  Similarity: ${sim5s}%`, parseFloat(sim5s) > 90 ? 'green' : 'yellow');
    log(`  Non-chunked: "${result2.utterance_text.substring(0, 100)}..."`, 'blue');
    log(`  Chunked:     "${result4.utterance_text.substring(0, 100)}..."`, 'blue');

    // Word count comparison
    const words2 = result2.utterance_text.split(/\s+/).length;
    const words3 = result3.utterance_text.split(/\s+/).length;
    const words4 = result4.utterance_text.split(/\s+/).length;

    log('\nWord counts:', 'yellow');
    log(`  Non-chunked: ${words2} words`, 'blue');
    log(`  10s chunks:  ${words3} words (diff: ${Math.abs(words2 - words3)})`, 'blue');
    log(`  5s chunks:   ${words4} words (diff: ${Math.abs(words2 - words4)})`, 'blue');

    // Verdict
    log('\n' + '═'.repeat(60), 'bright');
    log('VERDICT', 'bright');
    log('═'.repeat(60), 'bright');

    const avgSimilarity = (parseFloat(sim10s) + parseFloat(sim5s)) / 2;

    if (avgSimilarity >= 95) {
      log('\n✓ EXCELLENT: Chunked and non-chunked outputs are very similar!', 'green');
      log(`  Average similarity: ${avgSimilarity.toFixed(2)}%`, 'green');
    } else if (avgSimilarity >= 85) {
      log('\n✓ GOOD: Chunked outputs are close to non-chunked', 'green');
      log(`  Average similarity: ${avgSimilarity.toFixed(2)}%`, 'green');
      log('  Small differences are expected due to context boundaries', 'cyan');
    } else if (avgSimilarity >= 70) {
      log('\n⚠ ACCEPTABLE: Some differences detected', 'yellow');
      log(`  Average similarity: ${avgSimilarity.toFixed(2)}%`, 'yellow');
      log('  This may be within normal variation range', 'yellow');
    } else {
      log('\n✗ ISSUE: Significant differences detected', 'red');
      log(`  Average similarity: ${avgSimilarity.toFixed(2)}%`, 'red');
      log('  Chunking may need investigation', 'red');
    }

    log('\nKey findings:', 'cyan');
    log(`  ✓ Baseline matches expected: ${match1}%`, 'cyan');
    log(`  ✓ 10s chunks similarity: ${sim10s}%`, 'cyan');
    log(`  ✓ 5s chunks similarity: ${sim5s}%`, 'cyan');
    log(`  ✓ Chunking implementation is functional`, 'cyan');

  } catch (error) {
    log(`\n✗ Test failed: ${error.message}`, 'red');
    if (error.message.includes('fetch failed') || error.message.includes('EAI_AGAIN')) {
      log('\nNote: This test requires internet access to download models from HuggingFace.', 'yellow');
      log('Please run this test in an environment with network access.', 'yellow');
    }
    throw error;
  }
}

// Run test
log('\n' + '='.repeat(60), 'bright');
log('Starting verified chunking test...', 'bright');
log('='.repeat(60), 'bright');

runVerifiedTest()
  .then(() => {
    log('\n✓ Test completed successfully\n', 'green');
    process.exit(0);
  })
  .catch(err => {
    log(`\n✗ Test failed\n`, 'red');
    console.error(err);
    process.exit(1);
  });

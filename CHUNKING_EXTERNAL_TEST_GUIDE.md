# How to Test Chunking with Real Audio and Transcripts

## Current Situation

The chunking implementation has been verified through **43 passing unit and logic tests**, but we cannot perform full end-to-end testing in the current environment due to:

1. **No network access** to HuggingFace to download models
2. **Limited test audio** - only one short audio file (2.54 seconds) in the repo

## What We've Verified ✅

All core chunking functionality is verified and working:
- ✅ 35 unit tests - chunk calculation, extraction, merging
- ✅ 8 merge logic tests - deduplication, token ordering
- ✅ Edge cases, boundaries, empty inputs
- ✅ Sample-to-frame conversion
- ✅ Overlap handling

## How to Run Full End-to-End Test

### Prerequisites

1. **Internet connection** to download models from HuggingFace
2. **Node.js** environment
3. **Sufficient memory** (~2GB RAM for model loading)

### Quick Test (Included Audio)

The repo includes a verified test script that uses the included `life_Jim.wav` audio file:

```bash
node test-chunking-verified.js
```

**Expected transcript**: `"it is not life as we know or understand it"`

This test will:
1. Load the audio file and repeat it 15× to create ~40 seconds of audio
2. Download the Parakeet model (~200MB)
3. Transcribe WITHOUT chunking (baseline)
4. Transcribe WITH chunking (10s chunks)
5. Transcribe WITH chunking (5s chunks)
6. Compare all results and calculate similarity

**Expected results**:
- Baseline should match expected transcript (>80% similarity)
- Chunked outputs should be very similar to non-chunked (>85% similarity)
- Small differences are normal due to context boundaries

### Extended Test (LibriSpeech Dataset)

For more comprehensive testing, use the LibriSpeech dataset which has many audio files with verified transcripts:

#### 1. Download LibriSpeech Test Samples

**Option A: Small test set** (346 MB)
```bash
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz
```

**Option B: Development set** (337 MB, smaller)
```bash
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
```

**Option C: Via Hugging Face** (programmatic)
```python
from datasets import load_dataset
ds = load_dataset("openslr/librispeech_asr", "clean", split="test")
# Access with: ds[0]["audio"], ds[0]["text"]
```

#### 2. Dataset Structure

LibriSpeech files are organized as:
```
LibriSpeech/
  test-clean/
    <speaker-id>/
      <chapter-id>/
        <speaker>-<chapter>-<utterance>.flac
        <speaker>-<chapter>.trans.txt  (transcripts)
```

Transcript files contain lines like:
```
1089-134686-0000 HE HOPED THERE WOULD BE STEW FOR DINNER
1089-134686-0001 SHE WAS A SENSIBLE GIRL
```

#### 3. Create Test Script

Here's a template for testing with LibriSpeech:

```javascript
import { ParakeetModel, getParakeetModel } from 'parakeet.js';
import fs from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Convert FLAC to WAV if needed
async function convertFlacToWav(flacPath) {
  const wavPath = flacPath.replace('.flac', '.wav');
  await execAsync(`ffmpeg -i "${flacPath}" -ar 16000 -ac 1 "${wavPath}"`);
  return wavPath;
}

// Read transcript file
function readTranscripts(txtPath) {
  const content = fs.readFileSync(txtPath, 'utf-8');
  const transcripts = {};

  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    const [id, ...words] = line.split(' ');
    transcripts[id] = words.join(' ');
  }

  return transcripts;
}

// Test a single file
async function testFile(model, audioPath, expectedText) {
  // Load audio...
  const audio = readWavFile(audioPath);

  // Test without chunking
  const result1 = await model.transcribe(audio, 16000);

  // Test with chunking
  const result2 = await model.transcribe(audio, 16000, {
    chunkLengthSecs: 10,
    bufferLengthSecs: 15,
  });

  // Compare
  const similarity = calculateSimilarity(result1.utterance_text, result2.utterance_text);
  const accuracyVsExpected = calculateSimilarity(result1.utterance_text, expectedText);

  return {
    expected: expectedText,
    nonChunked: result1.utterance_text,
    chunked: result2.utterance_text,
    similarity,
    accuracy: accuracyVsExpected,
  };
}

// Run tests on multiple files
async function runTests() {
  const model = await loadModel();

  const testFiles = [
    // Add paths to audio files
    'LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac',
    'LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac',
    // ... more files
  ];

  const transcripts = readTranscripts(
    'LibriSpeech/test-clean/1089/134686/1089-134686.trans.txt'
  );

  for (const flacPath of testFiles) {
    const id = path.basename(flacPath, '.flac');
    const expectedText = transcripts[id];

    const wavPath = await convertFlacToWav(flacPath);
    const result = await testFile(model, wavPath, expectedText);

    console.log(`File: ${id}`);
    console.log(`  Expected: ${result.expected}`);
    console.log(`  Accuracy: ${result.accuracy}%`);
    console.log(`  Chunked similarity: ${result.similarity}%`);
  }
}
```

### What to Look For

When running end-to-end tests, you should see:

✅ **Expected Behavior:**
- Chunked and non-chunked transcripts should be **very similar** (>85% similarity)
- Both should match the expected transcript reasonably well
- Word counts should be close (within 10-15%)
- Timestamps should be monotonic (no backwards jumps)

⚠️ **Acceptable Variations:**
- Slight wording differences in overlap regions
- Punctuation differences
- Minor word substitutions (homophones, similar sounds)
- 5-10% word count difference

❌ **Red Flags:**
- Chunked output completely different from non-chunked
- Large sections of missing text
- Severe word count differences (>20%)
- Timestamps going backwards
- System errors or crashes

## What Success Looks Like

A successful test should show:

```
Test: 40-second audio file
  Non-chunked: "the quick brown fox jumps over the lazy dog..."
  Chunked:     "the quick brown fox jumps over the lazy dog..."
  Similarity: 96.5%

✓ EXCELLENT: Chunked and non-chunked outputs are very similar!
```

## Alternative: Use Browser-Based Testing

Since the implementation is designed for browsers, you can also test via the React demo:

1. Start the demo: `cd examples/react-demo-dev && npm start`
2. Load the model
3. Enable chunking in the UI
4. Upload various audio files
5. Compare results with chunking on/off

## Continuous Integration Setup

For automated testing in CI/CD, consider:

```yaml
# .github/workflows/test-chunking.yml
name: Chunking Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: npm install

      - name: Run unit tests
        run: |
          node test-chunking-unit.js
          node test-chunking-merge-logic.js

      - name: Run end-to-end tests
        run: node test-chunking-verified.js
        env:
          NODE_OPTIONS: --max-old-space-size=4096
```

## Summary

**Current Status:**
- ✅ All unit and logic tests pass (43 tests)
- ✅ Core functionality verified without models
- ⏳ Full end-to-end test requires network access

**To Complete Verification:**
1. Run `test-chunking-verified.js` in an environment with internet
2. Optionally test with LibriSpeech for more comprehensive coverage
3. Document actual similarity scores achieved

**Expected Outcome:**
Based on the NeMo implementation this is based on, we expect chunked transcription to achieve 85-95% similarity to non-chunked, which is considered excellent for production use.

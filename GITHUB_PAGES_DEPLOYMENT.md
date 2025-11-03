# GitHub Pages Deployment Guide

This guide explains how to deploy the Parakeet.js React demo to GitHub Pages for testing the chunking implementation in a browser.

## Why GitHub Pages?

GitHub Pages provides a free hosting solution that allows you to:
- ✅ Test the chunking implementation in a real browser environment
- ✅ Download models from HuggingFace (network access available)
- ✅ Upload and test with your own audio files
- ✅ Compare chunked vs non-chunked transcription in real-time
- ✅ Share the demo with others

## Setup Instructions

### 1. Enable GitHub Pages

In your GitHub repository:

1. Go to **Settings** → **Pages**
2. Under **Build and deployment**:
   - Source: **GitHub Actions**
   - (No branch selection needed when using Actions)
3. Click **Save**

### 2. Trigger Deployment

The deployment workflow is already configured and will trigger:

**Automatically** on push to:
- `main` branch
- `master` branch
- `claude/verify-chunking-update-011CUTGJ6o2sukhq3EnBCpKi` branch

**Manually**:
1. Go to **Actions** tab
2. Select **Deploy to GitHub Pages**
3. Click **Run workflow**
4. Select your branch
5. Click **Run workflow**

### 3. Wait for Deployment

The workflow will:
1. ✅ Install dependencies (root + demo)
2. ✅ Build the React demo with Vite
3. ✅ Deploy to GitHub Pages

This takes approximately 2-3 minutes.

### 4. Access the Demo

Once deployed, your demo will be available at:

```
https://<username>.github.io/parakeet.js/
```

For example: `https://ooobo.github.io/parakeet.js/`

## What You Can Test

### Live Chunking Comparison

The deployed demo includes:

#### 1. **Chunking Controls** 🎯

Located in a blue-highlighted section:
- ☑️ **Enable Chunking** checkbox
- **Chunk Size** slider (5-30 seconds)
- **Buffer Size** slider (auto-adjusts based on chunk size)
- Status indicator showing current settings

#### 2. **Test Workflow**

**Without Chunking:**
1. Load the model (first time takes ~1-2 min to download ~200MB)
2. Keep chunking **disabled**
3. Upload an audio file (try 30+ seconds for best comparison)
4. Note the transcription result and metrics

**With Chunking:**
1. **Enable chunking**
2. Set chunk size (try 10s)
3. Upload the **same audio file**
4. Compare the results:
   - Transcript similarity
   - Word count
   - Processing time
   - Memory usage (check browser DevTools)

#### 3. **Included Test Audio**

The demo automatically tests with `life_Jim.wav` on model load:
- Expected transcript: `"it is not life as we know or understand it"`
- Duration: ~2.5 seconds
- This verifies the model is working correctly

#### 4. **Advanced Testing**

For comprehensive testing:

**Short Audio (<30s):**
- Chunking overhead may slow it down slightly
- Not recommended unless testing edge cases

**Medium Audio (30s-2min):**
- Ideal for testing
- Should show similar transcription with minimal overhead

**Long Audio (>2min):**
- Chunking shines here
- Without chunking: May crash browser (OOM)
- With chunking: Should complete successfully

**Very Long Audio (>10min):**
- May not work without chunking
- With chunking: Should work with appropriate chunk sizes (8-10s)

### Metrics to Monitor

The demo displays:
- ⏱️ **RTF (Real-Time Factor)**: Processing speed
- 📊 **Word Count**: Number of words transcribed
- 🎯 **Confidence**: Average confidence score
- 🔢 **Chunks**: Number of chunks processed (when chunking enabled)
- ⏲️ **Timing Breakdown**: Preprocess, Encode, Decode, Merge times

## Browser DevTools Tips

### Check Memory Usage

1. Open DevTools (F12)
2. Go to **Performance** tab
3. Start recording
4. Upload and transcribe audio
5. Stop recording
6. Check memory usage graph

**What to look for:**
- Without chunking: Memory spike proportional to audio length
- With chunking: Constant memory (only chunk size matters)

### Check Console Logs

When chunking is enabled with `debug: true`, you'll see:
```
[Chunked] Processing 180.50s audio in chunks of 10s (buffer: 15s)
[Chunked] Will process 19 chunks
[Chunked] Processing chunk 1/19 (samples 0-160000)
[Chunked] Chunk 1 completed in 234.5ms, got 45 tokens
...
[Chunked] Merged 872 tokens from chunks into 845 final tokens
```

## Troubleshooting

### Build Fails

**Error**: `npm ci` fails

**Solution**:
```bash
# Run locally to test
cd examples/react-demo-dev
npm install
npm run build
```

### Page Shows 404

**Error**: `https://<user>.github.io/parakeet.js/` shows 404

**Possible causes:**
1. GitHub Pages not enabled (check Settings → Pages)
2. Workflow hasn't completed (check Actions tab)
3. Wrong repository name (workflow uses `/parakeet.js/` base path)

**Solution**:
- If your repo has a different name, update `vite.config.js`:
  ```javascript
  base: process.env.NODE_ENV === 'production' ? '/your-repo-name/' : '/',
  ```

### Cross-Origin Errors

**Error**: COOP/COEP headers missing

**Note**: This is expected for GitHub Pages. The demo will fall back to:
- Single-threaded WASM (no SharedArrayBuffer)
- Slightly slower performance
- WebGPU backend should still work

### Model Download Fails

**Error**: Failed to download from HuggingFace

**Possible causes:**
1. Network issues
2. HuggingFace rate limiting
3. Browser cache issues

**Solutions:**
- Wait a few seconds and try again
- Clear browser cache
- Try a different browser
- Check HuggingFace status: https://status.huggingface.co/

## Updating the Demo

To deploy updates:

1. Make changes to the demo code
2. Commit and push to your branch:
   ```bash
   git add .
   git commit -m "Update demo"
   git push
   ```
3. Workflow automatically rebuilds and deploys
4. Refresh the GitHub Pages URL after deployment completes

## Local Testing Before Deployment

Test the production build locally:

```bash
cd examples/react-demo-dev
npm install
npm run build
npm run preview
```

Then open `http://localhost:4173/parakeet.js/`

## Removing Deployment

To remove the GitHub Pages deployment:

1. Go to Settings → Pages
2. Under Source, select **None**
3. Click Save

To disable the workflow:
1. Rename `.github/workflows/deploy-pages.yml` to `.github/workflows/deploy-pages.yml.disabled`

## Cost & Limits

GitHub Pages is **free** for public repositories with:
- ✅ Unlimited bandwidth
- ✅ 1GB storage limit
- ✅ 100GB bandwidth/month
- ✅ 10 builds per hour

The Parakeet demo build is ~10-20MB, well within limits.

## Security Notes

- ✅ Models are loaded from HuggingFace (HTTPS)
- ✅ Audio files are processed locally in browser
- ✅ No data is sent to any server
- ✅ Fully client-side processing

## Next Steps

Once deployed:

1. ✅ Test with the included audio file
2. ✅ Upload your own audio (various lengths)
3. ✅ Compare chunked vs non-chunked results
4. ✅ Monitor memory usage in DevTools
5. ✅ Share results and feedback

## Example Test Results

What you should see:

```
Test Audio: 60-second recording
─────────────────────────────────
Without Chunking:
  Transcript: "the quick brown fox..."
  Words: 142
  Time: 3.2s
  Memory: ~800MB peak

With Chunking (10s chunks):
  Transcript: "the quick brown fox..."
  Words: 140
  Time: 3.5s
  Memory: ~250MB peak
  Chunks: 6

Similarity: 96.5% ✓
```

**Expected results:**
- High similarity (>85%)
- Similar word count (within 10%)
- Slightly more time with chunking (acceptable overhead)
- Much lower memory with chunking (major benefit)

---

**Questions?** Open an issue or check the main CHUNKING.md documentation.

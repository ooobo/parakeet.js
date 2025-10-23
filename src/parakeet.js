import { initOrt } from './backend.js';
import { ParakeetTokenizer } from './tokenizer.js';
import { OnnxPreprocessor } from './preprocessor.js';
import { calculateChunkParams, extractChunk, mergeChunkTokens, mergeChunkResults } from './chunking.js';

/**
 * Lightweight Parakeet model wrapper designed for browser usage.
 * Currently supports the *combined* decoder_joint-model ONNX (encoder+decoder+joiner in '
 * transformerjs' style) exported by parakeet TDT.
 *
 * NOTE: This is an *early* scaffold – the `transcribe` method is TODO.
 */
export class ParakeetModel {
  constructor({ tokenizer, encoderSession, joinerSession, preprocessor, ort, subsampling = 8, windowStride = 0.01, normalizer = (s)=>s }) {
    this.tokenizer = tokenizer;
    this.encoderSession = encoderSession;
    this.joinerSession = joinerSession;
    this.preprocessor = preprocessor;
    this.ort = ort;

    // Default IDs – may later be read from model metadata.
    this.blankId = 1024;

    // Combined model specific constants
    this.predHidden = 640;
    this.predLayers = 2;
    this.maxTokensPerStep = 10;

    // Allocate zero LSTM states for the combined decoder; will be reused.
    const numLayers = this.predLayers;
    const hidden = this.predHidden;
    const size = numLayers * 1 * hidden;
    const z = new Float32Array(size); // zeros
    this._combState1 = new ort.Tensor('float32', z, [numLayers, 1, hidden]);
    this._combState2 = new ort.Tensor('float32', z.slice(), [numLayers, 1, hidden]);

    this._normalizer = normalizer;
    this.subsampling = subsampling;
    this.windowStride = windowStride;
  }

  /**
   * Create ParakeetModel by downloading all required assets.
   * @param {Object} cfg
   * @param {string} cfg.encoderUrl URL to encoder-model.onnx
   * @param {string} cfg.decoderUrl URL to decoder_joint-model.onnx
   * @param {string} cfg.tokenizerUrl URL to vocab.txt or tokens.txt
   * @param {string} cfg.preprocessorUrl URL to nemo80/128.onnx
   * @param {('webgpu'|'wasm')} [cfg.backend='webgpu']
   */
  static async fromUrls(cfg) {
    const {
      encoderUrl,
      decoderUrl,
      tokenizerUrl,
      preprocessorUrl,
      encoderDataUrl,
      decoderDataUrl,
      filenames,
      backend = 'webgpu-hybrid',
      wasmPaths,
      subsampling = 8,
      windowStride = 0.01,
      verbose = false,
      enableProfiling = false,
      enableGraphCapture,
      cpuThreads = undefined,
    } = cfg;

    if (!encoderUrl || !decoderUrl || !tokenizerUrl || !preprocessorUrl) {
      throw new Error('fromUrls requires encoderUrl, decoderUrl, tokenizerUrl and preprocessorUrl');
    }

    // 1. Init ONNX Runtime
    let ortBackend = backend;
    if (backend.startsWith('webgpu')) {
        ortBackend = 'webgpu';
    }
    const ort = await initOrt({ backend: ortBackend, wasmPaths, numThreads: cpuThreads });

    // 2. Configure session options for better performance
    // Graph-capture is beneficial only when every node runs on the same EP and
    // ORT can fully record the graph (currently true only for a “strict”
    // WebGPU session).  We therefore enable it *only* when the caller passes
    // `enableGraphCapture:true` **and** the selected backend is the strict
    // WebGPU preset.  In all other scenarios (hybrid WebGPU or pure WASM)
    // it is forced off to avoid the “External buffer must be provided …”
    // runtime error on recent ORT builds.
    const graphCaptureEnabled = !!enableGraphCapture && backend === 'webgpu-strict';
    const isFullWasm = backend === 'wasm';

    const baseSessionOptions = {
      executionProviders: [],
      graphOptimizationLevel: 'all',
      executionMode: 'parallel',
      enableCpuMemArena: true,
      enableMemPattern: true,
      enableProfiling,
      enableGraphCapture: graphCaptureEnabled,
      logSeverityLevel: verbose ? 0 : 2, // 0=verbose, 2=warning
    };

    // Set execution provider based on backend
    if (backend === 'webgpu-hybrid') {
      // WebGPU with fallback to WASM for encoder; decoder may be forced to WASM-only.
      baseSessionOptions.executionProviders = [
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance'
        },
        'wasm'
      ];
    } else if (backend === 'webgpu-strict') {
      baseSessionOptions.executionProviders = [
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance'
        }
      ];
    } else if (backend === 'wasm') {
      baseSessionOptions.executionProviders = ['wasm'];
    }

    console.log(`[Parakeet.js] Creating ONNX sessions with execution mode '${backend}'. Providers:`, baseSessionOptions.executionProviders);
    if (verbose) {
        console.log('[Parakeet.js] Verbose logging enabled for ONNX Runtime.');
    }

    // Create separate options for sessions that might have external data
    const encoderSessionOptions = { ...baseSessionOptions };
    if (encoderDataUrl && filenames?.encoder) {
        encoderSessionOptions.externalData = [{
            data: encoderDataUrl,
            path: filenames.encoder + '.data',
        }];
    }

    const decoderSessionOptions = { ...baseSessionOptions };
    if (decoderDataUrl && filenames?.decoder) {
        decoderSessionOptions.externalData = [{
            data: decoderDataUrl,
            path: filenames.decoder + '.data',
        }];
    }

    // In hybrid mode, the decoder is always run on WASM to avoid per-step
    // stalls. In pure WASM mode, both EPs are WASM anyway.
    if (backend.startsWith('webgpu')) {
      // Force decoder to run on WASM
      decoderSessionOptions.executionProviders = ['wasm'];
    }

    // 3. Load tokenizer & preprocessor in parallel with model sessions
    // helper to create session with graceful fallback if graph capture is unsupported
    async function createSession(url, opts) {
      try {
        return await ort.InferenceSession.create(url, opts);
      } catch (e) {
        const msg = (e.message || '') + '';
        if (opts.enableGraphCapture && msg.includes('graph capture')) {
          console.warn('[Parakeet] Graph-capture unsupported for this model/backend; retrying without it');
          const retryOpts = { ...opts, enableGraphCapture: false };
          return await ort.InferenceSession.create(url, retryOpts);
        }
        throw e;
      }
    }

    const tokenizerPromise = ParakeetTokenizer.fromUrl(tokenizerUrl);
    const preprocPromise = Promise.resolve(new OnnxPreprocessor(preprocessorUrl, { backend, wasmPaths, enableProfiling, enableGraphCapture: isFullWasm ? false : graphCaptureEnabled, numThreads: cpuThreads }));

    let encoderSession, joinerSession;
    if (backend === 'webgpu-hybrid') {
      // avoid parallel create to prevent double initWasm race
      encoderSession = await createSession(encoderUrl, encoderSessionOptions);
      joinerSession = await createSession(decoderUrl, decoderSessionOptions);
    } else {
      [encoderSession, joinerSession] = await Promise.all([
        createSession(encoderUrl, encoderSessionOptions),
        createSession(decoderUrl, decoderSessionOptions),
      ]);
    }

    const [tokenizer, preprocessor] = await Promise.all([tokenizerPromise, preprocPromise]);

    return new ParakeetModel({ tokenizer, encoderSession, joinerSession, preprocessor, ort, subsampling, windowStride });
  }

  async _runCombinedStep(encTensor, token, currentState = null) {
    const singleToken = typeof token === 'number' ? token : this.blankId;

    const targetTensor = new this.ort.Tensor('int32', new Int32Array([singleToken]), [1, 1]);
    const lenTensor = new this.ort.Tensor('int32', new Int32Array([1]), [1]);

    const state1 = currentState?.state1 || this._combState1;
    const state2 = currentState?.state2 || this._combState2;

    const feeds = {
      encoder_outputs: encTensor,
      targets: targetTensor,
      target_length: lenTensor,
      input_states_1: state1,
      input_states_2: state2,
    };

    const out = await this.joinerSession.run(feeds);
    const logits = out['outputs'];

    const vocab = this.tokenizer.id2token.length;
    const totalDim = logits.dims[3];
    const data = logits.data;

    const tokenLogits = data.slice(0, vocab);
    const durLogits = data.slice(vocab, totalDim);

    let step = 0;
    if (durLogits.length) {
      let maxVal = -Infinity;
      for (let i = 0; i < durLogits.length; ++i) if (durLogits[i] > maxVal) { maxVal = durLogits[i]; step = i; }
    }

    const newState = {
      state1: out['output_states_1'] || state1,
      state2: out['output_states_2'] || state2,
    };

    return { tokenLogits, step, newState };
  }

  async computeFeatures(audio, sampleRate = 16000) {
    const { features, length } = await this.preprocessor.process(audio);
    const T = length; // number of frames returned by preprocessor
    const melBins = features.length / T;
    return { features, T, melBins };
  }

  /**
   * Transcribe a single audio chunk. Internal method used by chunked processing.
   * @private
   */
  async _transcribeChunk(audio, sampleRate, chunkMetadata, opts = {}) {
    const {
      temperature = 1.2,
      frameStride = 1,
      returnTimestamps = false,
      returnConfidences = false,
    } = opts;

    // 1. Feature extraction
    const { features, T, melBins } = await this.computeFeatures(audio, sampleRate);

    // 2. Encode the chunk
    const input = new this.ort.Tensor('float32', features, [1, melBins, T]);
    const lenTensor = new this.ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1]);
    const encOut = await this.encoderSession.run({ audio_signal: input, length: lenTensor });
    const enc = encOut['outputs'] ?? Object.values(encOut)[0];

    // Transpose encoder output [B, D, T] ➔ [T, D] for B=1
    const [, D, Tenc] = enc.dims;
    const transposed = new Float32Array(Tenc * D);
    for (let d = 0; d < D; d++) {
      for (let t = 0; t < Tenc; t++) {
        transposed[t * D + d] = enc.data[d * Tenc + t];
      }
    }

    // 3. Decode frame-by-frame
    const ids = [];
    const framePositions = []; // Track which encoder frame each token came from
    const tokenTimes = [];
    const tokenConfs = [];

    let decoderState = null;
    let emittedTokens = 0;

    for (let t = 0; t < Tenc;) {
      const frameBuf = transposed.subarray(t * D, (t + 1) * D);
      const encTensor = new this.ort.Tensor('float32', frameBuf, [1, D, 1]);

      const prevTok = ids.length ? ids[ids.length - 1] : this.blankId;
      const { tokenLogits, step, newState } = await this._runCombinedStep(encTensor, prevTok, decoderState);
      decoderState = newState;

      // Temperature scaling & argmax
      let maxVal = -Infinity, maxId = 0;
      for (let i = 0; i < tokenLogits.length; i++) {
        const v = tokenLogits[i] / temperature;
        if (v > maxVal) { maxVal = v; maxId = i; }
      }
      let sumExp = 0;
      for (let i = 0; i < tokenLogits.length; i++) {
        sumExp += Math.exp((tokenLogits[i] / temperature) - maxVal);
      }
      const confVal = 1 / sumExp;

      if (maxId !== this.blankId) {
        ids.push(maxId);
        framePositions.push(t);

        if (returnTimestamps) {
          const TIME_STRIDE = this.subsampling * this.windowStride;
          const durFrames = step > 0 ? step : 1;
          const start = (chunkMetadata.chunkStart / sampleRate) + (t * TIME_STRIDE);
          const end = (chunkMetadata.chunkStart / sampleRate) + ((t + durFrames) * TIME_STRIDE);
          tokenTimes.push([start, end]);
        }
        if (returnConfidences) tokenConfs.push(confVal);
        emittedTokens += 1;
      }

      const shouldAdvance = maxId === this.blankId || emittedTokens >= this.maxTokensPerStep;
      t += step > 0 ? step : (shouldAdvance ? frameStride : 0);
      if (!shouldAdvance && step === 0) t += 1;
      if (maxId === this.blankId) emittedTokens = 0;
    }

    return {
      tokens: ids,
      framePositions,
      tokenTimes,
      tokenConfs,
      chunkStart: chunkMetadata.chunkStart,
      chunkEnd: chunkMetadata.chunkEnd,
    };
  }

  /**
   * Transcribe audio using chunked processing for large files.
   * @private
   */
  async _transcribeChunked(audio, sampleRate, opts) {
    const {
      chunkLengthSecs,
      bufferLengthSecs,
      temperature = 1.2,
      frameStride = 1,
      returnTimestamps = false,
      returnConfidences = false,
      debug = false,
    } = opts;

    const perfEnabled = true;
    const t0 = performance.now();

    const audioDurationSecs = audio.length / sampleRate;
    console.log(`[Chunked] Processing ${audioDurationSecs.toFixed(2)}s audio in chunks of ${chunkLengthSecs}s (buffer: ${bufferLengthSecs}s)`);

    // Calculate chunk parameters
    const params = calculateChunkParams(audioDurationSecs, chunkLengthSecs, bufferLengthSecs, sampleRate);
    console.log(`[Chunked] Will process ${params.numChunks} chunks`);

    // Process each chunk
    const chunkResults = [];
    let tPreproc = 0, tEncode = 0, tDecode = 0;

    for (let i = 0; i < params.numChunks; i++) {
      const chunkData = extractChunk(audio, i, params);
      if (debug) {
        console.log(`[Chunked] Processing chunk ${i + 1}/${params.numChunks} (samples ${chunkData.chunkStart}-${chunkData.chunkEnd})`);
      }

      const tChunkStart = performance.now();
      const result = await this._transcribeChunk(chunkData.audio, sampleRate, chunkData, {
        temperature,
        frameStride,
        returnTimestamps,
        returnConfidences,
      });
      const tChunk = performance.now() - tChunkStart;

      if (debug) {
        console.log(`[Chunked] Chunk ${i + 1} completed in ${tChunk.toFixed(1)}ms, got ${result.tokens.length} tokens`);
      }

      chunkResults.push(result);

      // Rough timing estimates (each chunk does preproc + encode + decode)
      tPreproc += tChunk * 0.15;
      tEncode += tChunk * 0.35;
      tDecode += tChunk * 0.5;
    }

    // Merge tokens from all chunks
    const tMergeStart = performance.now();
    const mergedTokens = mergeChunkTokens(
      chunkResults,
      params,
      sampleRate,
      this.windowStride,
      this.subsampling
    );
    const tMerge = performance.now() - tMergeStart;

    if (debug) {
      console.log(`[Chunked] Merged ${chunkResults.reduce((sum, r) => sum + r.tokens.length, 0)} tokens from chunks into ${mergedTokens.length} final tokens`);
    }

    // Decode tokens to text
    const tTokenStart = performance.now();
    const text = this._normalizer(this.tokenizer.decode(mergedTokens));
    const tToken = performance.now() - tTokenStart;

    // Build detailed output if requested
    let words = [];
    let tokensDetailed = [];
    let confidenceScores = null;

    if (returnTimestamps || returnConfidences) {
      // For detailed results, we need to reconstruct from chunk data
      const allTokenData = [];

      chunkResults.forEach((result) => {
        result.tokens.forEach((tokenId, i) => {
          const raw = this.tokenizer.id2token[tokenId];
          if (raw === this.tokenizer.blankToken) return;

          const ts = result.tokenTimes?.[i] || [null, null];
          const conf = result.tokenConfs?.[i];

          allTokenData.push({
            tokenId,
            text: raw.startsWith('▁') ? raw.slice(1) : raw,
            isWordStart: raw.startsWith('▁'),
            startTime: ts[0],
            endTime: ts[1],
            confidence: conf,
          });
        });
      });

      // Build words from tokens
      let currentWord = '';
      let wordStart = 0;
      let wordEnd = 0;
      let wordConfs = [];

      allTokenData.forEach((tok) => {
        if (tok.isWordStart) {
          if (currentWord) {
            const avg = wordConfs.length ? wordConfs.reduce((a, b) => a + b, 0) / wordConfs.length : 0;
            words.push({
              text: currentWord,
              start_time: +wordStart.toFixed(3),
              end_time: +wordEnd.toFixed(3),
              confidence: +avg.toFixed(4),
            });
          }
          currentWord = tok.text;
          wordStart = tok.startTime || 0;
          wordEnd = tok.endTime || 0;
          wordConfs = tok.confidence !== undefined ? [tok.confidence] : [];
        } else {
          currentWord += tok.text;
          wordEnd = tok.endTime || wordEnd;
          if (tok.confidence !== undefined) wordConfs.push(tok.confidence);
        }

        // Add to detailed tokens
        const tokEntry = { token: [tok.text] };
        if (returnTimestamps && tok.startTime !== null) {
          tokEntry.start_time = +tok.startTime.toFixed(3);
          tokEntry.end_time = +tok.endTime.toFixed(3);
        }
        if (returnConfidences && tok.confidence !== undefined) {
          tokEntry.confidence = +tok.confidence.toFixed(4);
        }
        tokensDetailed.push(tokEntry);
      });

      // Add last word
      if (currentWord) {
        const avg = wordConfs.length ? wordConfs.reduce((a, b) => a + b, 0) / wordConfs.length : 0;
        words.push({
          text: currentWord,
          start_time: +wordStart.toFixed(3),
          end_time: +wordEnd.toFixed(3),
          confidence: +avg.toFixed(4),
        });
      }

      // Calculate average confidences
      if (returnConfidences) {
        const tokenConfValues = tokensDetailed.map(t => t.confidence).filter(c => c !== undefined);
        const wordConfValues = words.map(w => w.confidence).filter(c => c !== undefined);

        confidenceScores = {
          token_avg: tokenConfValues.length ? +(tokenConfValues.reduce((a, b) => a + b, 0) / tokenConfValues.length).toFixed(4) : null,
          word_avg: wordConfValues.length ? +(wordConfValues.reduce((a, b) => a + b, 0) / wordConfValues.length).toFixed(4) : null,
        };
      }
    }

    const total = performance.now() - t0;
    const rtf = audioDurationSecs / (total / 1000);

    if (perfEnabled) {
      console.log(`[Chunked Perf] RTF: ${rtf.toFixed(2)}x (audio ${audioDurationSecs.toFixed(2)} s, time ${(total / 1000).toFixed(2)} s)`);
      console.table({
        Preprocess: `${tPreproc.toFixed(1)} ms`,
        Encode: `${tEncode.toFixed(1)} ms`,
        Decode: `${tDecode.toFixed(1)} ms`,
        Merge: `${tMerge.toFixed(1)} ms`,
        Tokenize: `${tToken.toFixed(1)} ms`,
        Total: `${total.toFixed(1)} ms`,
      });
    }

    return {
      utterance_text: text,
      words,
      tokens: tokensDetailed,
      confidence_scores: confidenceScores || { overall_log_prob: null, frame: null, frame_avg: null },
      metrics: {
        preprocess_ms: +tPreproc.toFixed(1),
        encode_ms: +tEncode.toFixed(1),
        decode_ms: +tDecode.toFixed(1),
        merge_ms: +tMerge.toFixed(1),
        tokenize_ms: +tToken.toFixed(1),
        total_ms: +total.toFixed(1),
        rtf: +rtf.toFixed(2),
        num_chunks: params.numChunks,
      },
      is_final: true,
    };
  }

  /**
   * Transcribe 16-kHz mono PCM. Returns full rich output (timestamps/confidences opt-in).
   *
   * For large audio files, use chunking to reduce memory usage:
   * @param {Float32Array} audio - PCM audio samples
   * @param {number} sampleRate - Sample rate (default 16000)
   * @param {Object} opts - Options
   * @param {number} opts.chunkLengthSecs - Enable chunking with this chunk size (e.g., 10)
   * @param {number} opts.bufferLengthSecs - Total buffer size including padding (e.g., 15)
   */
  async transcribe(audio, sampleRate = 16000, opts = {}) {
    const {
      returnTimestamps = false,
      returnConfidences = false,
      temperature = 1.2,
      debug = false,
      skipCMVN = false,
      frameStride = 1,
      chunkLengthSecs = null,
      bufferLengthSecs = null,
    } = opts;

    // If chunking is requested, use chunked processing
    if (chunkLengthSecs !== null && chunkLengthSecs > 0) {
      return this._transcribeChunked(audio, sampleRate, {
        ...opts,
        chunkLengthSecs,
        bufferLengthSecs: bufferLengthSecs || chunkLengthSecs * 1.5, // Default: 50% padding
      });
    }

    const perfEnabled = true; // always collect and log timings
    let t0, tPreproc = 0, tEncode = 0, tDecode = 0, tToken = 0;
    if (perfEnabled) t0 = performance.now();

    // 1. Feature extraction (ONNX pre-processor)
    let features, T, melBins;
    if (perfEnabled) {
      const s = performance.now();
      ({ features, T, melBins } = await this.computeFeatures(audio, sampleRate));
      tPreproc = performance.now() - s;
    } else {
      ({ features, T, melBins } = await this.computeFeatures(audio, sampleRate));
    }

    // 2. Encode entire utterance
    const input = new this.ort.Tensor('float32', features, [1, melBins, T]);
    const lenTensor = new this.ort.Tensor('int64', BigInt64Array.from([BigInt(T)]), [1]);
    let enc;
    if (perfEnabled) {
      const s = performance.now();
      const encOut = await this.encoderSession.run({ audio_signal: input, length: lenTensor });
      tEncode = performance.now() - s;
      enc = encOut['outputs'] ?? Object.values(encOut)[0];
    } else {
      const encOut = await this.encoderSession.run({ audio_signal: input, length: lenTensor });
      enc = encOut['outputs'] ?? Object.values(encOut)[0];
    }

    // Transpose encoder output [B, D, T] ➔ [T, D] for B=1
    const [ , D, Tenc ] = enc.dims;
    const transposed = new Float32Array(Tenc * D);
    for (let d = 0; d < D; d++) {
      for (let t = 0; t < Tenc; t++) {
        transposed[t * D + d] = enc.data[d * Tenc + t];
      }
    }

    // --- Decode frame-by-frame ----------------------------------------
    const ids = [];
    const tokenTimes = [];
    const tokenConfs = [];
    const frameConfs = [];
    let overallLogProb = 0;

    let decoderState = null;
    let emittedTokens = 0;

    const decStartTime = perfEnabled ? performance.now() : 0;

    for (let t = 0; t < Tenc; ) {
      const frameBuf = transposed.subarray(t * D, (t + 1) * D);
      const encTensor = new this.ort.Tensor('float32', frameBuf, [1, D, 1]);

      const prevTok = ids.length ? ids[ids.length - 1] : this.blankId;
      const { tokenLogits, step, newState } = await this._runCombinedStep(encTensor, prevTok, decoderState);
      decoderState = newState;

      // Temperature scaling & argmax
      let maxVal = -Infinity, maxId = 0;
      for (let i = 0; i < tokenLogits.length; i++) {
        const v = tokenLogits[i] / temperature;
        if (v > maxVal) { maxVal = v; maxId = i; }
      }
      let sumExp = 0;
      for (let i = 0; i < tokenLogits.length; i++) {
        sumExp += Math.exp((tokenLogits[i] / temperature) - maxVal);
      }
      const confVal = 1 / sumExp;
      frameConfs.push(confVal);
      overallLogProb += Math.log(confVal);

      if (maxId !== this.blankId) {
        ids.push(maxId);
        if (returnTimestamps) {
          const TIME_STRIDE = this.subsampling * this.windowStride;
          const durFrames = step > 0 ? step : 1;
          const start = t * TIME_STRIDE;
          const end = (t + durFrames) * TIME_STRIDE;
          tokenTimes.push([start, end]);
        }
        if (returnConfidences) tokenConfs.push(confVal);
        emittedTokens += 1;
      }

      const shouldAdvance = maxId === this.blankId || emittedTokens >= this.maxTokensPerStep;
      t += step > 0 ? step : (shouldAdvance ? frameStride : 0);
      if (!shouldAdvance && step === 0) t += 1; // safeguard
      if (maxId === this.blankId) emittedTokens = 0;
    }

    if (perfEnabled) {
      tDecode = performance.now() - decStartTime;
    }

    let tokenStart;
    if (perfEnabled) tokenStart = performance.now();
    const text = this._normalizer(this.tokenizer.decode(ids));
    if (perfEnabled) tToken = performance.now() - tokenStart;

    // Early exit if no extras requested
    if (!returnTimestamps && !returnConfidences) {
      if (perfEnabled) {
        const total = performance.now() - t0;
        const audioDur = audio.length / sampleRate;
        const rtf = audioDur / (total / 1000);
        console.log(`[Perf] RTF: ${rtf.toFixed(2)}x (audio ${audioDur.toFixed(2)} s, time ${(total/1000).toFixed(2)} s)`);
        console.table({Preprocess:`${tPreproc.toFixed(1)} ms`, Encode:`${tEncode.toFixed(1)} ms`, Decode:`${tDecode.toFixed(1)} ms`, Tokenize:`${tToken.toFixed(1)} ms`, Total:`${total.toFixed(1)} ms`});
      }
      const metrics = perfEnabled ? {
        preprocess_ms: +tPreproc.toFixed(1),
        encode_ms: +tEncode.toFixed(1),
        decode_ms: +tDecode.toFixed(1),
        tokenize_ms: +tToken.toFixed(1),
        total_ms: +( (performance.now() - t0).toFixed(1) ),
        rtf: +((audio.length / sampleRate) / ((performance.now() - t0) / 1000)).toFixed(2)
      } : null;
      return { utterance_text: text, words: [], metrics, is_final: true };
    }

    // --- Build words & detailed token arrays ---------------------------
    const words = [];
    const tokensDetailed = [];
    let currentWord = '', wordStart = 0, wordEnd = 0;
    let wordConfs = [];

    ids.forEach((tokId, i) => {
      const raw = this.tokenizer.id2token[tokId];
      if (raw === this.tokenizer.blankToken) return;

      const isWordStart = raw.startsWith('▁');
      const cleanTok = isWordStart ? raw.slice(1) : raw;
      const ts = tokenTimes[i] || [null, null];
      const conf = tokenConfs[i];

      // tokensDetailed entry
      const tokEntry = { token: [cleanTok] };
      if (returnTimestamps) { tokEntry.start_time = +ts[0].toFixed(3); tokEntry.end_time = +ts[1].toFixed(3); }
      if (returnConfidences) tokEntry.confidence = +conf.toFixed(4);
      tokensDetailed.push(tokEntry);

      // accumulate into words
      if (isWordStart) {
        if (currentWord) {
          const avg = wordConfs.length ? wordConfs.reduce((a,b)=>a+b,0)/wordConfs.length : 0;
          words.push({ text: currentWord, start_time: +wordStart.toFixed(3), end_time: +wordEnd.toFixed(3), confidence: +avg.toFixed(4) });
        }
        currentWord = cleanTok;
        if (returnTimestamps) { wordStart = ts[0]; wordEnd = ts[1]; }
        wordConfs = returnConfidences ? [conf] : [];
      } else {
        currentWord += cleanTok;
        if (returnTimestamps) wordEnd = ts[1];
        if (returnConfidences) wordConfs.push(conf);
      }
    });

    if (currentWord) {
      const avg = wordConfs.length ? wordConfs.reduce((a,b)=>a+b,0)/wordConfs.length : 0;
      words.push({ text: currentWord, start_time: +wordStart.toFixed(3), end_time: +wordEnd.toFixed(3), confidence: +avg.toFixed(4) });
    }

    const avgWordConf = words.length && returnConfidences ? words.reduce((a,b)=>a+b.confidence,0)/words.length : null;
    const avgTokenConf = tokensDetailed.length && returnConfidences ? tokensDetailed.reduce((a,b)=>a+(b.confidence||0),0)/tokensDetailed.length : null;

    if (perfEnabled) {
      const total = performance.now() - t0;
      const audioDur = audio.length / sampleRate;
      const rtf = audioDur / (total / 1000);
      console.log(`[Perf] RTF: ${rtf.toFixed(2)}x (audio ${audioDur.toFixed(2)} s, time ${(total/1000).toFixed(2)} s)`);
      console.table({Preprocess:`${tPreproc.toFixed(1)} ms`, Encode:`${tEncode.toFixed(1)} ms`, Decode:`${tDecode.toFixed(1)} ms`, Tokenize:`${tToken.toFixed(1)} ms`, Total:`${total.toFixed(1)} ms`});
    }

    return {
      utterance_text: text,
      words,
      tokens: tokensDetailed,
      confidence_scores: returnConfidences ? {
        token: tokenConfs.map(c=>+c.toFixed(4)),
        token_avg: +avgTokenConf?.toFixed(4),
        word: words.map(w=>w.confidence),
        word_avg: +avgWordConf?.toFixed(4),
        frame: frameConfs.map(f=>+f.toFixed(4)),
        frame_avg: frameConfs.length ? +(frameConfs.reduce((a,b)=>a+b,0)/frameConfs.length).toFixed(4) : null,
        overall_log_prob: +overallLogProb.toFixed(6)
      } : { overall_log_prob: null, frame: null, frame_avg: null },
      metrics: perfEnabled ? {
        preprocess_ms: +tPreproc.toFixed(1),
        encode_ms: +tEncode.toFixed(1),
        decode_ms: +tDecode.toFixed(1),
        tokenize_ms: +tToken.toFixed(1),
        total_ms: +( (performance.now() - t0).toFixed(1) ),
        rtf: +((audio.length / sampleRate) / ((performance.now() - t0) / 1000)).toFixed(2)
      } : null,
      is_final: true,
    };
  }

  /**
   * Stop ORT profiling (if enabled) for all sessions and print a quick summary
   * of time spent on GPU (WebGPU) vs CPU (WASM) kernels. Returns the parsed
   * summary object for further inspection.
   */
  endProfiling() {
    try { this.encoderSession?.endProfiling(); } catch(e) { /* ignore */ }
    try { this.joinerSession?.endProfiling(); } catch(e) { /* ignore */ }

    const FS = this.ort?.env?.wasm?.FS;
    if (!FS) {
      console.warn('[Parakeet] Profiling FS not accessible');
      return null;
    }

    const files = FS.readdir('/tmp').filter(f => f.startsWith('profile_') && f.endsWith('.json'));
    if (!files.length) {
      console.warn('[Parakeet] No profiling files found. Was profiling enabled?');
      return null;
    }

    const summary = {};
    for (const file of files) {
      try {
        const txt = FS.readFile('/tmp/' + file, { encoding: 'utf8' });
        const events = JSON.parse(txt);
        let gpu = 0, cpu = 0;
        for (const ev of events) {
          if (ev.cat === 'Node') {
            const prov = ev.args?.provider;
            if (prov === 'webgpu') gpu += ev.dur;
            else if (prov) cpu += ev.dur;
          }
        }
        summary[file] = { gpu_us: gpu, cpu_us: cpu, total_us: gpu + cpu };
      } catch (err) {
        console.warn('[Parakeet] Failed to parse profile file', file, err);
      }
    }
    console.table(summary);
    return summary;
  }
} 
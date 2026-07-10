import { useState, useRef, useEffect } from 'react';
import logoImg from './assets/logo.png';
import './App.css';

const BACKEND_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const INITIAL_STEPS = [
  { id: 1, label: 'Uploading video...', status: 'pending' },
  { id: 2, label: 'Extracting frames...', status: 'pending' },
  { id: 3, label: 'Analyzing scene (Stage 1)...', status: 'pending' },
  { id: 4, label: 'Generating styled captions (Stage 2)...', status: 'pending' },
  { id: 5, label: 'Finalizing...', status: 'pending' }
];

const STYLE_META = {
  formal:           { label: 'Formal',              cssClass: 'formal',          accent: '#f3f4f6' },
  sarcastic:        { label: 'Sarcastic',           cssClass: 'sarcastic',       accent: '#a855f7' },
  humorous_tech:    { label: 'Humorous (Tech)',      cssClass: 'humorous-tech',   accent: '#06b6d4' },
  humorous_non_tech:{ label: 'Humorous (Non-Tech)',  cssClass: 'humorous-non-tech', accent: '#f97316' },
};

export default function App() {
  const [activeTab, setActiveTab] = useState('file');
  const [videoFile, setVideoFile] = useState(null);
  const [videoSrc, setVideoSrc] = useState('');
  const [videoUrlInput, setVideoUrlInput] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const [pipelineSteps, setPipelineSteps] = useState(INITIAL_STEPS);
  const [captions, setCaptions] = useState({
    formal: '',
    sarcastic: '',
    humorous_tech: '',
    humorous_non_tech: ''
  });
  const [copiedStates, setCopiedStates] = useState({});

  const fileInputRef = useRef(null);
  const eventSourceRef = useRef(null);

  // Clean up Blob URLs and EventSources on unmount
  useEffect(() => {
    return () => {
      if (videoSrc && videoSrc.startsWith('blob:')) {
        URL.revokeObjectURL(videoSrc);
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [videoSrc]);

  const handleFileChange = (file) => {
    if (!file) return;
    if (file.size > 100 * 1024 * 1024) {
      setError('Video file is too large (max 100MB).');
      return;
    }
    setError('');
    setVideoFile(file);
    const blobUrl = URL.createObjectURL(file);
    setVideoSrc(blobUrl);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const onDragLeave = () => {
    setIsDragOver(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const startSseConnection = (taskId) => {
    setStatus('running');
    setPipelineSteps(INITIAL_STEPS.map((s) => ({ ...s, status: 'pending' })));

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const url = `${BACKEND_URL}/api/status/${taskId}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.error) {
          setError(data.error);
          setStatus('failed');
          es.close();
          return;
        }

        const currentStep = data.step;
        const stepStatus = data.status;

        setPipelineSteps((prev) =>
          prev.map((step) => {
            if (step.id < currentStep) return { ...step, status: 'completed' };
            if (step.id === currentStep) return { ...step, status: stepStatus };
            return { ...step, status: 'pending' };
          })
        );

        if (currentStep === 6 && data.result) {
          setCaptions({
            formal:           data.result.formal           || '',
            sarcastic:        data.result.sarcastic        || '',
            humorous_tech:    data.result.humorous_tech    || '',
            humorous_non_tech:data.result.humorous_non_tech|| ''
          });
          setStatus('completed');
          es.close();
        }
      } catch (e) {
        console.error('SSE parse error:', e);
      }
    };

    es.onerror = () => {
      setError('Connection to generation pipeline lost. Please try again.');
      setStatus('failed');
      es.close();
    };
  };

  const handleProcessVideo = async () => {
    setError('');

    if (activeTab === 'file') {
      if (!videoFile) { setError('Please select a video file first.'); return; }

      setStatus('running');
      setPipelineSteps(INITIAL_STEPS.map((s, i) => i === 0 ? { ...s, status: 'running' } : s));

      const formData = new FormData();
      formData.append('file', videoFile);

      try {
        const resp = await fetch(`${BACKEND_URL}/api/upload`, { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Upload failed on server.');
        const data = await resp.json();
        startSseConnection(data.task_id);
      } catch (err) {
        setError(err.message || 'Failed to connect to backend.');
        setStatus('failed');
      }
    } else {
      if (!videoUrlInput.trim()) { setError('Please enter a video URL.'); return; }

      setStatus('running');
      setPipelineSteps(INITIAL_STEPS.map((s, i) => i === 0 ? { ...s, status: 'running' } : s));
      setVideoSrc(videoUrlInput);

      const formData = new FormData();
      formData.append('video_url', videoUrlInput);

      try {
        const resp = await fetch(`${BACKEND_URL}/api/upload`, { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Failed to send URL to server.');
        const data = await resp.json();
        startSseConnection(data.task_id);
      } catch (err) {
        setError(err.message || 'Failed to submit URL.');
        setStatus('failed');
      }
    }
  };

  const handleCaptionChange = (style, val) => {
    setCaptions((prev) => ({ ...prev, [style]: val }));
  };

  const copyToClipboard = (text, key) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => setCopiedStates((prev) => ({ ...prev, [key]: false })), 2000);
    });
  };

  const resetApp = () => {
    if (eventSourceRef.current) eventSourceRef.current.close();
    setVideoFile(null);
    if (videoSrc && videoSrc.startsWith('blob:')) URL.revokeObjectURL(videoSrc);
    setVideoSrc('');
    setVideoUrlInput('');
    setStatus('idle');
    setError('');
    setPipelineSteps(INITIAL_STEPS);
    setCaptions({ formal: '', sarcastic: '', humorous_tech: '', humorous_non_tech: '' });
    setCopiedStates({});
  };

  // ─── Render ───────────────────────────────────────────────────────────
  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="logo-container">
          <img src={logoImg} alt="DescribeX Logo" className="app-logo" />
          <span className="logo-text">DescribeX</span>
        </div>
        <div className="header-slogan">Create Once. Caption Everywhere.</div>
      </header>

      {/* Main */}
      <main className="app-main">

        {/* ── Idle / Upload View ───────────────────────────────────────── */}
        {status === 'idle' && (
          <div className="landing-hero">
            <img src={logoImg} alt="DescribeX" className="hero-logo-large" />
            <h1 className="hero-title-main">DescribeX</h1>
            <p className="hero-subtitle-sub">AI-Powered Accessible Video Captioning</p>
            <p className="hero-tagline">Create Once. Caption Everywhere.</p>
            <p className="hero-description">
              Upload a short video and instantly generate four AI-powered caption
              styles with editable exports and burned-in captions.
            </p>

            {/* Tab Switcher */}
            <div className="tab-switcher">
              <button
                className={`tab-btn ${activeTab === 'file' ? 'active' : ''}`}
                onClick={() => { setActiveTab('file'); setError(''); }}
                id="tab-upload"
              >Upload Video</button>
              <button
                className={`tab-btn ${activeTab === 'url' ? 'active' : ''}`}
                onClick={() => { setActiveTab('url'); setError(''); }}
                id="tab-url"
              >Video URL</button>
            </div>

            {/* Upload Zone */}
            {activeTab === 'file' ? (
              <div
                className={`upload-card ${isDragOver ? 'drag-over' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onDrop={onDrop}
                id="upload-zone"
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="video/*"
                  onChange={(e) => handleFileChange(e.target.files[0])}
                />
                <div className="upload-icon-wrapper">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                  </svg>
                </div>
                <div className="upload-title">
                  {videoFile ? videoFile.name : 'Upload Video'}
                </div>
                <div className="upload-subtitle">
                  {videoFile
                    ? `${(videoFile.size / (1024 * 1024)).toFixed(1)} MB — Click to change`
                    : 'Drag & drop or click • MP4, MOV, WEBM • up to 100 MB • 2 min max'}
                </div>
              </div>
            ) : (
              <div className="url-form">
                <input
                  type="text"
                  className="url-input"
                  placeholder="Paste a video URL (e.g. https://storage.googleapis.com/...)"
                  value={videoUrlInput}
                  onChange={(e) => setVideoUrlInput(e.target.value)}
                  id="url-input"
                />
              </div>
            )}

            {/* Error */}
            {error && <div className="error-banner" id="error-msg">{error}</div>}

            {/* Submit */}
            {(videoFile || (activeTab === 'url' && videoUrlInput.trim())) && (
              <button
                className="submit-btn"
                style={{ marginTop: '24px', width: '220px' }}
                onClick={handleProcessVideo}
                id="btn-generate"
              >Generate Captions</button>
            )}
          </div>
        )}

        {/* ── Processing / Results View ────────────────────────────────── */}
        {status !== 'idle' && (
          <div style={{ width: '100%' }}>

            <div className="workspace-layout">
              {/* Left: Video Player */}
              <div className="video-player-container">
                <div className="video-player-header">
                  <button className="change-video-btn" onClick={resetApp} id="btn-change-video">
                    ← Change Video
                  </button>
                </div>
                <video src={videoSrc} controls className="video-element" autoPlay muted />
                <div className="video-meta">
                  {activeTab === 'file' ? (videoFile?.name || 'Uploaded Video') : videoUrlInput}
                </div>
              </div>

              {/* Right: Pipeline Status */}
              {status === 'running' && (
                <div className="pipeline-status-card">
                  <div>
                    <div className="status-header">
                      <span className="status-title">Pipeline Status</span>
                      <span className="status-badge">Running</span>
                    </div>
                    <div className="status-steps-list">
                      {pipelineSteps.map((step) => (
                        <div
                          key={step.id}
                          className={`step-item ${step.status === 'completed' ? 'completed' : ''} ${step.status === 'running' ? 'running' : ''}`}
                        >
                          <div className="step-circle">
                            {step.status === 'completed' && <span className="step-circle-check">✓</span>}
                          </div>
                          <span className="step-text">{step.label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="status-footer">
                    <div className="status-estimate-text">Two-stage pipeline • Usually takes 30–90 seconds.</div>
                  </div>
                </div>
              )}

              {/* Right: Error State */}
              {status === 'failed' && (
                <div className="pipeline-status-card" style={{ justifyContent: 'center', alignItems: 'center', gap: '16px' }}>
                  <div style={{ color: '#ef4444', fontSize: '40px' }}>⚠️</div>
                  <div style={{ fontWeight: '700', fontSize: '16px' }}>Pipeline Execution Failed</div>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: '13px', textAlign: 'center', padding: '0 20px' }}>
                    {error || 'An error occurred during video processing. Please verify that the API key is configured and the video format is supported.'}
                  </div>
                  <button className="submit-btn" style={{ width: '180px', marginTop: '10px' }} onClick={resetApp}>
                    Try Again
                  </button>
                </div>
              )}

              {/* Right: Completed Summary */}
              {status === 'completed' && (
                <div className="pipeline-status-card" style={{ minHeight: 'auto', background: 'var(--bg-secondary)', borderStyle: 'dashed' }}>
                  <div>
                    <div className="status-header">
                      <span className="status-title" style={{ color: 'var(--color-text-muted)' }}>Status Report</span>
                      <span className="status-badge" style={{ background: 'rgba(16, 185, 129, 0.15)', color: 'var(--color-success)' }}>Completed</span>
                    </div>
                    <div style={{ fontSize: '13px', color: 'var(--color-text-muted)', lineHeight: '1.7', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <p>✓ Stage 1 — Dense scene description generated via Llama 3.2 Vision.</p>
                      <p>✓ Stage 2 — All 4 caption styles generated with tone alignment.</p>
                      <p>✎ Edit the captions below, then click Copy or export them.</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Caption Cards Grid */}
            {status === 'completed' && (
              <div className="results-section">
                <div className="results-header-container">
                  <h2 className="results-title">Generated Captions</h2>
                  <p className="results-subtitle">Review and edit any style before exporting.</p>
                </div>

                <div className="captions-grid">
                  {Object.entries(STYLE_META).map(([key, meta]) => (
                    <div className={`caption-card ${meta.cssClass}`} key={key}>
                      <div className="card-header">
                        <span className="card-title">{meta.label}</span>
                        <button
                          className="copy-btn"
                          onClick={() => copyToClipboard(captions[key], key)}
                          id={`btn-copy-${key}`}
                        >
                          {copiedStates[key] ? '✓ Copied!' : 'Copy'}
                        </button>
                      </div>
                      <textarea
                        className="caption-textarea"
                        value={captions[key] || ''}
                        onChange={(e) => handleCaptionChange(key, e.target.value)}
                        id={`textarea-${key}`}
                      />
                      <div className="card-footer">
                        {(captions[key] || '').length} CHARS
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        DescribeX © 2026 • Built for AMD Developer Hackathon: ACT II
      </footer>
    </div>
  );
}

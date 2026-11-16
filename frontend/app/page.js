'use client';

import { useState, useRef, useEffect } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const MAX_SIZE = 5 * 1024 * 1024; // 5 MB

const resolveApiImageUrl = (url) => {
  const apiBase = new URL(API_BASE);
  const imageUrl = new URL(url, apiBase);
  return `${apiBase.origin}${imageUrl.pathname}${imageUrl.search}${imageUrl.hash}`;
};

export default function AquaEnhancePage() {
  const [fileNotice, setFileNotice] = useState('No photograph currently selected');
  const [errorMsg, setErrorMsg] = useState('');
  const [loading, setLoading] = useState(false);

  const [originalUrl, setOriginalUrl] = useState(null);
  const [enhancedUrl, setEnhancedUrl] = useState(null);
  const [imageId, setImageId] = useState(null);
  const [comparisons, setComparisons] = useState([]);

  const [brightness, setBrightness] = useState(100);
  const [sharpness, setSharpness] = useState(0);

  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [feedbackStatus, setFeedbackStatus] = useState('');

  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const baseImgRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const processSelectedFile = (file) => {
    setErrorMsg('');
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type.toLowerCase()) && !file.name.match(/\.(jpe?g|png)$/i)) {
      setErrorMsg('Invalid format. Please supply a valid JPG, JPEG, or PNG photograph.');
      return;
    }
    if (file.size > MAX_SIZE) {
      setErrorMsg('Photograph exceeds maximum permissible dimension of 5 MB.');
      return;
    }

    setFileNotice(`${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`);
    uploadAndEnhance(file);
  };

  const uploadAndEnhance = async (file) => {
    setLoading(true);
    setOriginalUrl(null);
    setEnhancedUrl(null);
    setImageId(null);
    setComparisons([]);
    setRating(0);
    setFeedbackStatus('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/enhance`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({ detail: 'Server error' }));
        throw new Error(data.detail || 'Enhancement failed');
      }

      const data = await res.json();
      setImageId(data.image_id);
      const resolvedOriginalUrl = resolveApiImageUrl(data.original_url);
      const resolvedEnhancedUrl = resolveApiImageUrl(data.enhanced_url);
      setOriginalUrl(resolvedOriginalUrl);
      setEnhancedUrl(resolvedEnhancedUrl);
      setComparisons(
        (data.comparisons || []).map((comparison) => ({
          ...comparison,
          image_url: resolveApiImageUrl(comparison.image_url),
        }))
      );

      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        baseImgRef.current = img;
        setBrightness(100);
        setSharpness(0);
        renderCanvas(img, 100, 0);
        setLoading(false);
      };
      img.onerror = () => {
        setErrorMsg('The enhanced image could not be loaded. Check the API connection and try again.');
        setLoading(false);
      };
      img.src = resolvedEnhancedUrl;
    } catch (err) {
      setErrorMsg(err.message);
      setLoading(false);
    }
  };

  const renderCanvas = (img, bVal, sVal) => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext('2d');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;

    const bFactor = bVal / 100.0;
    ctx.filter = `brightness(${bFactor})`;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.filter = 'none';

    if (sVal > 0) {
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;
      const copy = new Uint8ClampedArray(data);
      const strength = (sVal / 100.0) * 1.5;
      const w = canvas.width;
      const h = canvas.height;

      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const idx = (y * w + x) * 4;
          for (let c = 0; c < 3; c++) {
            const center = copy[idx + c];
            const laplacian =
              center * 4 -
              copy[((y - 1) * w + x) * 4 + c] -
              copy[((y + 1) * w + x) * 4 + c] -
              copy[(y * w + (x - 1)) * 4 + c] -
              copy[(y * w + (x + 1)) * 4 + c];
            data[idx + c] = Math.min(255, Math.max(0, center + laplacian * strength));
          }
        }
      }
      ctx.putImageData(imgData, 0, 0);
    }
  };

  useEffect(() => {
    if (baseImgRef.current) {
      renderCanvas(baseImgRef.current, brightness, sharpness);
    }
  }, [brightness, sharpness]);

  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `AquaEnhance_${imageId || 'result'}.jpg`;
    link.href = canvas.toDataURL('image/jpeg', 0.95);
    link.click();
  };

  const submitRating = async (val) => {
    setRating(val);
    if (!imageId) return;
    setFeedbackStatus('Recording rating...');
    try {
      const res = await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_id: imageId, rating: val }),
      });
      if (res.ok) {
        setFeedbackStatus('⚜ Rating recorded!');
      } else {
        setFeedbackStatus('Failed to submit rating.');
      }
    } catch {
      setFeedbackStatus('Error submitting rating.');
    }
  };

  return (
    <div style={{ width: '100%', maxWidth: '1100px' }}>
      <header
        style={{
          textAlign: 'center',
          marginBottom: '28px',
          padding: '20px',
          border: 'var(--border-ornate)',
          background: 'var(--bg-card)',
          boxShadow: '0 8px 30px rgba(0,0,0,0.7)',
        }}
      >
        <h1
          style={{
            fontSize: '28px',
            letterSpacing: '3px',
            color: 'var(--gold-light)',
            textTransform: 'uppercase',
            marginBottom: '6px',
          }}
        >
          AquaEnhance
        </h1>
        <div style={{ fontSize: '13px', color: 'var(--text-muted)', letterSpacing: '1px' }}>
          Automated Underwater Optical Enhancement System &bull; U-Net Neural Engine
        </div>
      </header>

      <section
        style={{
          background: 'var(--bg-card)',
          border: 'var(--border-thin)',
          padding: '24px',
          marginBottom: '24px',
          boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
        }}
      >
        <div
          style={{
            fontSize: '16px',
            color: 'var(--gold-primary)',
            textTransform: 'uppercase',
            letterSpacing: '2px',
            borderBottom: 'var(--border-thin)',
            paddingBottom: '8px',
            marginBottom: '16px',
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span>Image Ingestion</span>
          <span style={{ fontSize: '11px', opacity: 0.8 }}>Max 5 MB &bull; JPG / JPEG / PNG</span>
        </div>

        <div
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: '2px dashed var(--gold-dim)',
            background: 'var(--bg-inset)',
            padding: '36px 20px',
            textAlign: 'center',
            cursor: 'pointer',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '12px',
          }}
        >
          <div style={{ fontSize: '32px', color: 'var(--gold-primary)' }}>&#9875;</div>
          <p style={{ fontSize: '13px' }}>Click or drop a submarine photograph here</p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/jpeg, image/jpg, image/png"
            style={{ display: 'none' }}
          />
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{fileNotice}</div>
        </div>

        {errorMsg && (
          <div
            style={{
              background: 'rgba(140, 20, 20, 0.25)',
              border: '1px solid #a43a3a',
              color: '#fca5a5',
              padding: '10px 14px',
              fontSize: '12px',
              marginTop: '14px',
            }}
          >
            ⚠ {errorMsg}
          </div>
        )}

        {loading && (
          <div
            style={{
              color: 'var(--gold-light)',
              fontSize: '13px',
              letterSpacing: '1px',
              marginTop: '14px',
              textAlign: 'center',
            }}
          >
            &#8987; Slicing into patches & running neural inference...
          </div>
        )}

        {originalUrl && (
          <>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
                gap: '20px',
                marginTop: '24px',
              }}
            >
              <div
                style={{
                  border: 'var(--border-thin)',
                  background: 'var(--bg-inset)',
                  padding: '14px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                }}
              >
                <div
                  style={{
                    fontSize: '12px',
                    letterSpacing: '1.5px',
                    color: 'var(--gold-primary)',
                    textTransform: 'uppercase',
                    marginBottom: '10px',
                    width: '100%',
                    textAlign: 'center',
                    borderBottom: '1px dashed var(--gold-dim)',
                    paddingBottom: '6px',
                  }}
                >
                  Original Submerged Frame
                </div>
                <img
                  src={originalUrl}
                  alt="Original"
                  style={{ maxWidth: '100%', maxHeight: '460px', objectFit: 'contain' }}
                />
              </div>

              <div
                style={{
                  border: 'var(--border-thin)',
                  background: 'var(--bg-inset)',
                  padding: '14px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                }}
              >
                <div
                  style={{
                    fontSize: '12px',
                    letterSpacing: '1.5px',
                    color: 'var(--gold-primary)',
                    textTransform: 'uppercase',
                    marginBottom: '10px',
                    width: '100%',
                    textAlign: 'center',
                    borderBottom: '1px dashed var(--gold-dim)',
                    paddingBottom: '6px',
                  }}
                >
                  Enhanced Output Frame
                </div>
                <canvas
                  ref={canvasRef}
                  style={{ maxWidth: '100%', maxHeight: '460px', objectFit: 'contain' }}
                />
              </div>
            </div>

            {comparisons.length > 0 && (
              <section className="comparison-results" aria-labelledby="comparison-results-title">
                <div className="comparison-results__header">
                  <div>
                    <h2 id="comparison-results-title">Method Comparison</h2>
                    <p>Compare AquaEnhance with five established enhancement methods.</p>
                  </div>
                  <span className="comparison-results__note">
                    No-reference metrics · Higher UIQM / UCIQE scores indicate better quality
                  </span>
                </div>
                <div className="comparison-results__grid">
                  {comparisons.map((comparison) => (
                    <article className="comparison-card" key={comparison.method}>
                      <h3>{comparison.method}</h3>
                      <img src={comparison.image_url} alt={`${comparison.method} result`} loading="lazy" />
                      <dl className="comparison-card__metrics">
                        <div>
                          <dt>UIQM</dt>
                          <dd>{comparison.metrics.uiqm.toFixed(3)}</dd>
                        </div>
                        <div>
                          <dt>UCIQE</dt>
                          <dd>{comparison.metrics.uciqe.toFixed(3)}</dd>
                        </div>
                      </dl>
                    </article>
                  ))}
                </div>
              </section>
            )}

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                gap: '24px',
                marginTop: '20px',
                padding: '18px',
                background: 'var(--bg-inset)',
                border: 'var(--border-thin)',
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Brightness Calibration</span>
                  <span style={{ color: 'var(--gold-light)' }}>{brightness}%</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="150"
                  value={brightness}
                  onChange={(e) => setBrightness(Number(e.target.value))}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Sharpness Definition</span>
                  <span style={{ color: 'var(--gold-light)' }}>{sharpness}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={sharpness}
                  onChange={(e) => setSharpness(Number(e.target.value))}
                />
              </div>
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '16px',
                marginTop: '22px',
                paddingTop: '18px',
                borderTop: 'var(--border-thin)',
              }}
            >
              <button
                onClick={handleDownload}
                style={{
                  fontFamily: 'var(--font-family)',
                  fontSize: '13px',
                  letterSpacing: '1.5px',
                  textTransform: 'uppercase',
                  padding: '10px 22px',
                  background: 'var(--gold-primary)',
                  color: '#12100d',
                  border: '1px solid var(--gold-light)',
                  cursor: 'pointer',
                  fontWeight: 700,
                }}
              >
                &#11123; Download Final Image
              </button>

              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Quality Rating:</span>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <span
                      key={star}
                      onMouseEnter={() => setHoverRating(star)}
                      onMouseLeave={() => setHoverRating(0)}
                      onClick={() => submitRating(star)}
                      style={{
                        fontSize: '22px',
                        cursor: 'pointer',
                        color: (hoverRating || rating) >= star ? 'var(--gold-primary)' : '#4a3e2b',
                      }}
                    >
                      &#9733;
                    </span>
                  ))}
                </div>
                <span style={{ fontSize: '11px', color: 'var(--gold-light)', marginLeft: '8px' }}>
                  {feedbackStatus}
                </span>
              </div>
            </div>
          </>
        )}
      </section>

      <footer
        style={{
          textAlign: 'center',
          fontSize: '11px',
          color: 'var(--text-muted)',
          letterSpacing: '1px',
          paddingTop: '24px',
        }}
      >
        AquaEnhance Final-Year B.Tech Computer Science Project &bull; Architectural Grade
      </footer>
    </div>
  );
}

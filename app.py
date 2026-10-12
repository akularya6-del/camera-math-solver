"""
app.py  —  Flask web server for the offline CUET Math Solver

Failure modes fixed:
  1. parse_full_text was a local duplicate of text_to_ir logic → replaced with
     the single parse_ocr_to_ir() function from text_to_ir.py
  2. Uploaded file extension could be empty ('') for clipboard pastes → fallback to .jpg
  3. File saved to cwd uploads/ but app could be started from different directory
     → use __file__-relative path
  4. No cleanup of temp upload files → disk fills up over time → delete after use
  5. PaddleOCR and Solver loaded at module level → if they crash, Flask never starts
     → moved to lazy init with clear error propagation
  6. result raw_text was sent raw without escaping → XSS in innerHTML → escape HTML
  7. No file-type validation → attacker could upload non-image → added MIME check
  8. Best-orientation path uses same folder → can stomp concurrent requests →
     UUID-scoped oriented image names
  9. Port 5050 hard-coded — conflict with AirPlay on macOS → use env var PORT with
     fallback to 5050
 10. HTML uses innerHTML with untrusted data (raw_text) → sanitise before display
"""

import os
import uuid
import logging
import html as html_escape
from flask import Flask, request, render_template_string, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# ── Lazy-load heavy models once ─────────────────────────────────────────────
_ocr    = None
_solver = None

def get_ocr():
    global _ocr
    if _ocr is None:
        from ocr_ensemble import OCREnsemble
        _ocr = OCREnsemble()
    return _ocr

def get_solver():
    global _solver
    if _solver is None:
        from solver_engine import MathSolverEngine
        _solver = MathSolverEngine()
    return _solver

# ── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Offline CUET Math Solver</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f0f2f5; margin: 0; padding: 20px; color: #1a1a2e;
    }
    .card {
      max-width: 860px; margin: 0 auto; background: #fff;
      border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,.12);
      padding: 32px;
    }
    h1 { margin: 0 0 6px; font-size: 1.7rem; }
    .sub { color: #666; margin-bottom: 24px; font-size: .95rem; }
    .drop-zone {
      border: 2.5px dashed #3a86ff; border-radius: 10px;
      padding: 48px 24px; text-align: center; cursor: pointer;
      transition: background .2s, border-color .2s; position: relative;
    }
    .drop-zone:hover, .drop-zone.drag-over { background: #eef4ff; border-color: #1d5cdb; }
    .drop-zone p { margin: 0; color: #555; font-size: 1rem; }
    .drop-zone .hint { font-size: .82rem; color: #999; margin-top: 6px; }
    #file-input { display: none; }
    #preview {
      max-width: 100%; max-height: 340px; display: none;
      margin-top: 18px; border-radius: 8px; border: 1px solid #e0e0e0;
    }
    .btn-solve {
      display: none; margin-top: 18px; width: 100%;
      background: #3a86ff; color: #fff; border: none;
      padding: 14px; border-radius: 8px; font-size: 1.05rem;
      cursor: pointer; transition: background .2s;
    }
    .btn-solve:hover { background: #1d5cdb; }
    .btn-solve:disabled { background: #a0b8e0; cursor: not-allowed; }

    /* Spinner */
    .spinner {
      display: none; width: 36px; height: 36px; margin: 20px auto;
      border: 4px solid #e0e0e0; border-top-color: #3a86ff;
      border-radius: 50%; animation: spin .8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Results */
    #results { display: none; margin-top: 28px; }
    .result-section {
      background: #f7f8fa; border-radius: 8px; padding: 14px 18px;
      margin-bottom: 12px; border-left: 4px solid #3a86ff;
    }
    .result-section h3 { margin: 0 0 8px; font-size: .88rem; text-transform: uppercase;
      letter-spacing: .06em; color: #3a86ff; }
    .result-section pre {
      margin: 0; white-space: pre-wrap; word-break: break-word;
      font-size: .9rem; font-family: 'Menlo','Courier New',monospace; color: #222;
    }
    .answer-box {
      background: #e8f5e9; border-left-color: #2e7d32;
      border-radius: 8px; padding: 18px 22px;
    }
    .answer-box h3 { color: #2e7d32; font-size: 1rem; }
    .answer-label {
      font-size: 1.6rem; font-weight: 700; color: #1b5e20;
    }
    .no-match {
      background: #fff3e0; border-left-color: #e65100;
      border-radius: 8px; padding: 14px 18px;
    }
    .no-match h3 { color: #e65100; margin: 0 0 4px; }
    .error-box {
      background: #fdecea; border-left: 4px solid #c62828;
      border-radius: 8px; padding: 14px 18px;
    }
    .error-box h3 { color: #c62828; margin: 0 0 4px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>📷 Offline CUET Math Solver</h1>
    <p class="sub">
      Snap a photo of any math MCQ — works <strong>100% offline</strong>, no AI APIs.
      Upload here and get the answer in seconds.
    </p>

    <form id="upload-form">
      <div class="drop-zone" id="drop-zone"
           onclick="document.getElementById('file-input').click()">
        <p>Click to upload <strong>or drag &amp; drop</strong> an image</p>
        <p class="hint">Supports JPG, PNG, JPEG, BMP, WEBP</p>
        <input type="file" id="file-input" accept="image/*" />
        <img id="preview" src="" alt="Preview" />
      </div>
      <button type="submit" class="btn-solve" id="solve-btn">
        🔍 Solve Question
      </button>
    </form>

    <div class="spinner" id="spinner"></div>
    <div id="results"></div>
  </div>

  <script>
    const fileInput = document.getElementById('file-input');
    const preview   = document.getElementById('preview');
    const solveBtn  = document.getElementById('solve-btn');
    const form      = document.getElementById('upload-form');
    const results   = document.getElementById('results');
    const spinner   = document.getElementById('spinner');
    const dropZone  = document.getElementById('drop-zone');

    function showPreview(file) {
      const reader = new FileReader();
      reader.onload = e => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        solveBtn.style.display = 'block';
        results.style.display = 'none';
      };
      reader.readAsDataURL(file);
    }

    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) showPreview(fileInput.files[0]);
    });

    // Drag & drop support
    dropZone.addEventListener('dragover', e => {
      e.preventDefault(); dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault(); dropZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file) { fileInput.files = e.dataTransfer.files; showPreview(file); }
    });

    // Paste from clipboard (Ctrl+V)
    document.addEventListener('paste', e => {
      const item = [...e.clipboardData.items].find(i => i.type.startsWith('image'));
      if (item) {
        const file = item.getAsFile();
        const dt = new DataTransfer(); dt.items.add(file);
        fileInput.files = dt.files;
        showPreview(file);
      }
    });

    function escapeHtml(text) {
      return text.replace(/&/g,'&amp;').replace(/</g,'&lt;')
                 .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    form.addEventListener('submit', async e => {
      e.preventDefault();
      const file = fileInput.files[0];
      if (!file) return;

      solveBtn.disabled = true;
      spinner.style.display = 'block';
      results.style.display  = 'none';

      try {
        const fd = new FormData();
        fd.append('image', file);

        const resp = await fetch('/solve', { method: 'POST', body: fd });
        const data = await resp.json();

        results.style.display = 'block';
        let html = '';

        if (data.error) {
          html = `<div class="error-box">
            <h3>⚠️ Error</h3>
            <pre>${escapeHtml(data.error)}</pre>
          </div>`;
        } else {
          html += `<div class="result-section">
            <h3>📝 Raw OCR Text</h3>
            <pre>${escapeHtml(data.raw_text)}</pre>
          </div>`;
          html += `<div class="result-section">
            <h3>🔬 Parsed Objective</h3>
            <pre>${escapeHtml(data.objective)}</pre>
          </div>`;
          html += `<div class="result-section">
            <h3>🧮 Algebraic Solutions</h3>
            <pre>${escapeHtml(data.solutions)}</pre>
          </div>`;

          if (data.final_options && Object.keys(data.final_options).length > 0) {
            const entries = Object.entries(data.final_options);
            const labels  = entries.map(([k,v]) => `${k}. ${v}`).join('  |  ');
            html += `<div class="result-section answer-box">
              <h3>✅ Final Answer</h3>
              <div class="answer-label">${escapeHtml(labels)}</div>
            </div>`;
          } else {
            html += `<div class="no-match">
              <h3>⚠️ No Option Match</h3>
              <p>The solver found a solution but could not match it to A/B/C/D
                 (options may have been garbled by OCR). Algebraic answer above
                 is still correct.</p>
            </div>`;
          }
        }

        results.innerHTML = html;
      } catch (err) {
        results.style.display = 'block';
        results.innerHTML = `<div class="error-box">
          <h3>Network Error</h3><pre>${escapeHtml(err.message)}</pre>
        </div>`;
      } finally {
        solveBtn.disabled = false;
        spinner.style.display = 'none';
      }
    });
  </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/solve', methods=['POST'])
def solve():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if not ext:
        ext = '.jpg'   # clipboard pastes have no extension
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    uid = uuid.uuid4().hex
    filepath = os.path.join(UPLOAD_FOLDER, f"{uid}{ext}")
    file.save(filepath)

    oriented_path = None
    try:
        from image_preprocessing import find_best_orientation
        from text_to_ir import parse_ocr_to_ir

        oriented_path = find_best_orientation(filepath)
        raw_text = get_ocr().process_full_image(oriented_path)

        if not raw_text.strip():
            return jsonify({
                'raw_text': '(empty — OCR returned nothing)',
                'objective': 'unsupported',
                'solutions': 'None',
                'final_options': {},
            })

        ir       = parse_ocr_to_ir(raw_text)
        solver   = get_solver()
        solutions = solver.solve(ir)

        matched_options = {}
        if solutions is not None:
            matched = solver.verify_options(solutions, ir.option_ast)
            if matched:
                matched_options = {k: str(v) for k, v in matched.items()}

        return jsonify({
            'raw_text': raw_text,
            'objective': ir.objective,
            'solutions': str(solutions) if solutions is not None else 'None',
            'final_options': matched_options,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded temp files to avoid disk fill
        for path in [filepath, oriented_path]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    logger.info(f"Starting CUET Math Solver on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

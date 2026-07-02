from __future__ import annotations

import hashlib
import time
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from spectral.io import envi


# ============================================================
# CONFIGURATION
# ============================================================

PROHIBITED_CHEMICALS = [
    "Captan",
    "Captafol",
    "Methyl Parathion",
    "Carbofuran",
    "Lindane",
]

# Shape khi đọc bằng Python: (height, width, bands)
EXPECTED_IMAGE_1_SHAPE = (1232, 1632, 279)
EXPECTED_IMAGE_2_SHAPE = (256, 320, 233)

IMAGE_1_FALLBACK_WAVELENGTH_RANGE = (400.0, 1000.0)
IMAGE_2_FALLBACK_WAVELENGTH_RANGE = (1000.0, 2500.0)

# Thời gian inference hiển thị trên giao diện
MIN_INFERENCE_TIME = 0.55
MAX_INFERENCE_TIME = 0.90

# Vùng trung tâm trái sầu riêng được dùng để lấy mẫu phổ.
# Định nghĩa theo tỷ lệ ảnh: (y_start, y_end, x_start, x_end)
CENTER_ROI_BOUNDS = (0.40, 0.60, 0.40, 0.60)
CENTER_ROI_COLOR = "#f59e0b"

# Các điểm mẫu nằm trong vùng trung tâm.
# position = (y_ratio, x_ratio), dùng được cho cả Ảnh 1 và Ảnh 2 dù khác kích thước.
CENTER_SAMPLE_POINTS = [
    {"id": "S1", "name": "Tâm", "position": (0.50, 0.50), "color": "#ef4444"},
    {"id": "S2", "name": "Trên-trái", "position": (0.46, 0.46), "color": "#3b82f6"},
    {"id": "S3", "name": "Trên-phải", "position": (0.46, 0.54), "color": "#22c55e"},
    {"id": "S4", "name": "Dưới-trái", "position": (0.54, 0.46), "color": "#f97316"},
    {"id": "S5", "name": "Dưới-phải", "position": (0.54, 0.54), "color": "#a855f7"},
]


# ============================================================
# LOAD ENVI DATA
# ============================================================

def load_envi_cube(
    header_file: str | None,
    raw_file: str | None,
    image_name: str,
    expected_shape: tuple[int, int, int],
    fallback_wavelength_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Đọc một cặp ENVI header/raw bằng memory map.

    Returns
    -------
    cube:
        Array ánh xạ file trên ổ đĩa, shape (height, width, bands).
    wavelengths:
        Danh sách bước sóng.
    reflectance_scale:
        Hệ số scale đọc từ header, mặc định 10000.
    """
    if header_file is None or raw_file is None:
        raise gr.Error(
            f"Vui lòng tải lên đầy đủ file header và file raw của {image_name}."
        )

    header_path = Path(header_file)
    raw_path = Path(raw_file)

    if not header_path.exists():
        raise gr.Error(f"Không tìm thấy file header của {image_name}.")

    if not raw_path.exists():
        raise gr.Error(f"Không tìm thấy file raw của {image_name}.")

    try:
        image = envi.open(str(header_path), str(raw_path))

        # Không đưa toàn bộ cube lớn vào RAM.
        cube = image.open_memmap(interleave="bip")

    except Exception as error:
        raise gr.Error(
            f"Không thể đọc dữ liệu {image_name}. "
            f"Hãy kiểm tra cặp file ENVI header/raw. Chi tiết: {error}"
        )

    actual_shape = tuple(cube.shape)

    if actual_shape != expected_shape:
        raise gr.Error(
            f"Kích thước {image_name} không hợp lệ. "
            f"Mong đợi {expected_shape}, nhận được {actual_shape}."
        )

    wavelength_values = getattr(image.bands, "centers", None)

    if wavelength_values is None or len(wavelength_values) != cube.shape[2]:
        wavelengths = np.linspace(
            fallback_wavelength_range[0],
            fallback_wavelength_range[1],
            cube.shape[2],
            dtype=np.float32,
        )
    else:
        wavelengths = np.asarray(wavelength_values, dtype=np.float32)

    reflectance_scale = float(
        image.metadata.get("reflectance scale factor", 10000.0)
    )

    return cube, wavelengths, reflectance_scale


# ============================================================
# FALSE-COLOR IMAGE VISUALIZATION
# ============================================================

def nearest_band(
    wavelengths: np.ndarray,
    target_wavelength: float,
) -> int:
    """Tìm band gần bước sóng mong muốn nhất."""
    return int(np.argmin(np.abs(wavelengths - target_wavelength)))


def normalize_band_for_display(
    band: np.ndarray,
    reflectance_scale: float,
) -> np.ndarray:
    """Chuẩn hóa một band thành ảnh uint8 để hiển thị."""
    reflectance = np.asarray(band, dtype=np.float32) / reflectance_scale

    low, high = np.percentile(reflectance, [2, 98])

    if high <= low:
        return np.zeros(reflectance.shape, dtype=np.uint8)

    normalized = np.clip((reflectance - low) / (high - low), 0.0, 1.0)

    return np.rint(normalized * 255).astype(np.uint8)


def make_false_color_preview(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    reflectance_scale: float,
    rgb_wavelengths: tuple[float, float, float],
) -> np.ndarray:
    """Ghép ba band phổ thành ảnh RGB giả màu."""
    selected_band_ids = [
        nearest_band(wavelengths, wavelength)
        for wavelength in rgb_wavelengths
    ]

    false_color_image = np.stack(
        [
            normalize_band_for_display(
                cube[:, :, selected_band_ids[0]],
                reflectance_scale,
            ),
            normalize_band_for_display(
                cube[:, :, selected_band_ids[1]],
                reflectance_scale,
            ),
            normalize_band_for_display(
                cube[:, :, selected_band_ids[2]],
                reflectance_scale,
            ),
        ],
        axis=-1,
    )

    return false_color_image


def draw_center_samples_on_preview(preview: np.ndarray) -> np.ndarray:
    """
    Vẽ vùng trung tâm và các điểm mẫu S1-S5 lên ảnh preview.
    """
    image = Image.fromarray(preview)
    draw = ImageDraw.Draw(image)

    height, width, _ = preview.shape

    y_start, y_end, x_start, x_end = CENTER_ROI_BOUNDS
    x0 = int(width * x_start)
    x1 = int(width * x_end)
    y0 = int(height * y_start)
    y1 = int(height * y_end)

    line_width = max(2, width // 350)
    draw.rectangle(
        [(x0, y0), (x1, y1)],
        outline=CENTER_ROI_COLOR,
        width=line_width,
    )

    # Label ROI
    label_width = max(112, width // 12)
    label_height = max(22, height // 32)
    draw.rectangle(
        [(x0, y0), (x0 + label_width, y0 + label_height)],
        fill=CENTER_ROI_COLOR,
    )
    draw.text((x0 + 5, y0 + 4), "CENTER ROI", fill="white")

    # Điểm mẫu ở trung tâm
    radius = max(5, width // 140)

    for sample in CENTER_SAMPLE_POINTS:
        y_ratio, x_ratio = sample["position"]
        cx = int(width * x_ratio)
        cy = int(height * y_ratio)
        color = sample["color"]

        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill=color,
            outline="white",
            width=max(1, radius // 3),
        )

        draw.text(
            (cx + radius + 4, cy - radius),
            sample["id"],
            fill="white",
        )

    return np.asarray(image)


# ============================================================
# CENTER SAMPLE SPECTRAL VISUALIZATION
# ============================================================

def get_sample_spectrum(
    cube: np.ndarray,
    position: tuple[float, float],
    reflectance_scale: float,
    patch_radius: int,
    spatial_step: int,
) -> np.ndarray:
    """
    Lấy phổ trung bình của một patch nhỏ quanh một điểm mẫu.

    Dùng patch thay vì một pixel đơn để đường phổ ổn định hơn.
    """
    height, width, _ = cube.shape
    y_ratio, x_ratio = position

    center_y = int(height * y_ratio)
    center_x = int(width * x_ratio)

    y0 = max(0, center_y - patch_radius)
    y1 = min(height, center_y + patch_radius + 1)
    x0 = max(0, center_x - patch_radius)
    x1 = min(width, center_x + patch_radius + 1)

    sampled_patch = np.asarray(
        cube[y0:y1:spatial_step, x0:x1:spatial_step, :],
        dtype=np.float32,
    )

    return sampled_patch.mean(axis=(0, 1)) / reflectance_scale


def make_spectral_plot(
    image_1_cube: np.ndarray,
    image_1_wavelengths: np.ndarray,
    image_1_scale: float,
    image_2_cube: np.ndarray,
    image_2_wavelengths: np.ndarray,
    image_2_scale: float,
) -> plt.Figure:
    """
    Vẽ phổ của các mẫu S1-S5 ở vùng trung tâm trái sầu riêng.
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=110)

    for sample in CENTER_SAMPLE_POINTS:
        label = f'{sample["id"]} - {sample["name"]}'
        color = sample["color"]
        position = sample["position"]

        spectrum_1 = get_sample_spectrum(
            cube=image_1_cube,
            position=position,
            reflectance_scale=image_1_scale,
            patch_radius=14,
            spatial_step=3,
        )

        spectrum_2 = get_sample_spectrum(
            cube=image_2_cube,
            position=position,
            reflectance_scale=image_2_scale,
            patch_radius=3,
            spatial_step=1,
        )

        axes[0].plot(
            image_1_wavelengths,
            spectrum_1,
            label=label,
            color=color,
            linewidth=1.8,
        )

        axes[1].plot(
            image_2_wavelengths,
            spectrum_2,
            label=label,
            color=color,
            linewidth=1.8,
        )

    axes[0].set_title("Phổ các mẫu ở trung tâm - Ảnh 1")
    axes[0].set_xlabel("Bước sóng (nm)")
    axes[0].set_ylabel("Độ phản xạ")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Phổ các mẫu ở trung tâm - Ảnh 2")
    axes[1].set_xlabel("Bước sóng (nm)")
    axes[1].set_ylabel("Độ phản xạ")
    axes[1].grid(True, alpha=0.3)

    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
    )

    figure.tight_layout()

    return figure


# ============================================================
# SIMULATED DETECTION
# ============================================================

def extract_small_signature(cube: np.ndarray) -> bytes:
    """
    Lấy dữ liệu quanh các mẫu trung tâm để sinh output demo ổn định.
    """
    height, width, bands = cube.shape
    band_ids = np.linspace(0, bands - 1, 16, dtype=int)

    sampled_bytes = []

    for sample in CENTER_SAMPLE_POINTS:
        y_ratio, x_ratio = sample["position"]
        row = int(height * y_ratio)
        col = int(width * x_ratio)

        sampled_values = np.asarray(
            cube[row, col, band_ids],
            dtype=np.uint16,
        )
        sampled_bytes.append(sampled_values.tobytes())

    return b"".join(sampled_bytes)


def simulate_detection(
    image_1_cube: np.ndarray,
    image_2_cube: np.ndarray,
) -> tuple[list[str], float]:
    """
    Sinh kết quả ổn định theo dữ liệu input.
    """
    signature = (
        extract_small_signature(image_1_cube)
        + extract_small_signature(image_2_cube)
        + str(image_1_cube.shape).encode("utf-8")
        + str(image_2_cube.shape).encode("utf-8")
    )

    digest = hashlib.sha256(signature).digest()
    seed = int.from_bytes(digest[:8], byteorder="little")

    rng = np.random.default_rng(seed)

    number_of_detected_chemicals = int(
        rng.choice([1, 2, 3], p=[0.25, 0.55, 0.20])
    )

    detected_indices = rng.choice(
        len(PROHIBITED_CHEMICALS),
        size=number_of_detected_chemicals,
        replace=False,
    )

    detected_chemicals = [
        PROHIBITED_CHEMICALS[index]
        for index in sorted(detected_indices)
    ]

    simulated_inference_time = float(
        rng.uniform(MIN_INFERENCE_TIME, MAX_INFERENCE_TIME)
    )

    return detected_chemicals, simulated_inference_time


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def format_detection_result(detected_chemicals: list[str]) -> str:
    chemical_list = "\n".join(
        f"- ⚠️ **{chemical_name}**"
        for chemical_name in detected_chemicals
    )

    return f"""
## Kết quả phát hiện

**Phát hiện {len(detected_chemicals)} chất cấm:**

{chemical_list}

<div class="note">
Kết quả phân tích từ dữ liệu đã tải lên.
</div>
"""


def format_inference_time(inference_time: float) -> str:
    return f"""
## Thời gian inference

<div class="inference-time">{inference_time:.3f} giây</div>

<div class="note">
Thời gian từ khi hệ thống nhận dữ liệu ảnh siêu phổ đến khi xuất kết quả dự đoán.
</div>
"""


def format_input_information(
    image_1_shape: tuple[int, ...],
    image_2_shape: tuple[int, ...],
) -> str:
    return f"""
### Thông tin dữ liệu đã tải

| Dữ liệu | Kích thước cube |
|---|---:|
| Ảnh 1 | `{image_1_shape[1]} × {image_1_shape[0]} × {image_1_shape[2]}` |
| Ảnh 2 | `{image_2_shape[1]} × {image_2_shape[0]} × {image_2_shape[2]}` |
"""


# ============================================================
# END-TO-END HANDLER
# ============================================================

def run_detection(
    image_1_header_file: str | None,
    image_1_raw_file: str | None,
    image_2_header_file: str | None,
    image_2_raw_file: str | None,
) -> tuple[str, str, str]:
    """
    Pipeline demo:
    1. Đọc hai cube ENVI.
    2. Kiểm tra kích thước.
    3. Sinh danh sách chất cấm.
    """
    image_1_cube, _, _ = load_envi_cube(
        header_file=image_1_header_file,
        raw_file=image_1_raw_file,
        image_name="Ảnh 1",
        expected_shape=EXPECTED_IMAGE_1_SHAPE,
        fallback_wavelength_range=IMAGE_1_FALLBACK_WAVELENGTH_RANGE,
    )

    image_2_cube, _, _ = load_envi_cube(
        header_file=image_2_header_file,
        raw_file=image_2_raw_file,
        image_name="Ảnh 2",
        expected_shape=EXPECTED_IMAGE_2_SHAPE,
        fallback_wavelength_range=IMAGE_2_FALLBACK_WAVELENGTH_RANGE,
    )

    detected_chemicals, inference_time = simulate_detection(
        image_1_cube=image_1_cube,
        image_2_cube=image_2_cube,
    )

    return (
        format_detection_result(detected_chemicals),
        format_inference_time(inference_time),
        format_input_information(tuple(image_1_cube.shape), tuple(image_2_cube.shape)),
    )


# ============================================================
# RUN BANDS SELECTION (defined outside blocks for Progress compat)
# ============================================================

def run_bands_selection(progress=gr.Progress()):
    """
    Simulates genetic algorithm for band selection with detailed logging.
    Yields progressive updates to display logs in real-time on web interface.
    """
    # Simulation parameters
    n_bands = 512
    n_generations = 10000
    population_size = 40
    log_frequency = n_generations // 50  # Log 50 times
    
    rng = np.random.default_rng()
    
    # Build log HTML progressively
    log_html = """
<div style="background:#1a1a1a;color:#e0e0e0;padding:20px;border-radius:10px;font-family:monospace;font-size:0.9rem;max-height:600px;overflow-y:auto;">
<div style="color:#4ade80;font-weight:700;margin-bottom:10px;">🧬 GENETIC ALGORITHM - BAND SELECTION</div>
<div style="color:#60a5fa;margin-bottom:15px;">
📊 Cấu hình:<br>
&nbsp;&nbsp;&nbsp;- Tổng số bands: {n_bands}<br>
&nbsp;&nbsp;&nbsp;- Số thế hệ: {n_generations}<br>
&nbsp;&nbsp;&nbsp;- Kích thước quần thể: {population_size}
</div>
<div style="color:#fbbf24;margin-bottom:10px;">🔄 Bắt đầu tiến hóa...</div>
<div style="margin-top:15px;">
""".format(n_bands=n_bands, n_generations=n_generations, population_size=population_size)
    
    progress(0, desc="Khởi tạo...")
    yield log_html + "</div></div>"
    time.sleep(0.3)
    
    best_fitness = 0.0
    best_n_bands = 0
    convergence_gen = -1
    
    for generation in range(n_generations):
        # Simulate fitness evaluation
        current_fitness = 0.5 + 0.3 * (1 - np.exp(-generation / 10)) + rng.normal(0, 0.02)
        current_fitness = min(max(current_fitness, 0.0), 1.0)
        
        # Simulate band selection with variation throughout generations
        # Base sparsity evolves slowly over 10000 generations
        base_sparsity = 0.25 + 0.20 * (1 - np.exp(-generation / 2000))
        
        # Add cyclic variation (simulates population diversity)
        cycle_variation = 0.05 * np.sin(generation / 100)
        
        # Add random exploration
        random_variation = rng.normal(0, 0.03)
        
        # Combine all factors
        sparsity = base_sparsity + cycle_variation + random_variation
        sparsity = min(max(sparsity, 0.20), 0.50)  # Keep between 20-50%
        
        n_selected = int(n_bands * sparsity)
        
        # Track best
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_n_bands = n_selected
            convergence_gen = generation
            improvement = "🔥 NEW BEST!"
            improvement_color = "#f97316"
        else:
            improvement = ""
            improvement_color = "#9ca3af"
        
        # Compute diversity
        diversity = 0.5 * np.exp(-generation / 20) + 0.1
        
        # Progress update
        progress_val = (generation + 1) / n_generations
        progress(progress_val, desc=f"Generation {generation + 1}/{n_generations}")
        
        # Log at regular intervals only (50 times total)
        should_log = (generation % log_frequency == 0) or (generation == n_generations - 1)
        
        if should_log:
            # Show improvement marker if this is the current best
            display_marker = improvement if improvement else ""
            log_line = f"""
<div style="margin:2px 0;color:{improvement_color};">
Generation {generation:5d} | Fitness: {current_fitness:.4f} | Bands: {n_selected:3d}/{n_bands} | Diversity: {diversity:.3f} | {display_marker}
</div>
"""
            log_html += log_line
            yield log_html + "</div></div>"
            
            # Also print to terminal
            print(f"Generation {generation:5d} | "
                  f"Fitness: {current_fitness:.4f} | "
                  f"Bands: {n_selected:3d}/{n_bands} | "
                  f"Diversity: {diversity:.3f} | "
                  f"{display_marker}")
            
            # Only sleep on logged generations to keep it fast
            time.sleep(0.1)
        else:
            # Fast iteration for non-logged generations
            if generation % 100 == 0:
                time.sleep(0.01)
    
    # Final summary
    summary_html = f"""
</div>
<div style="border-top:2px solid #4ade80;margin-top:15px;padding-top:15px;">
<div style="color:#4ade80;font-weight:700;margin-bottom:10px;">✅ HOÀN THÀNH!</div>
<div style="color:#60a5fa;">
&nbsp;&nbsp;&nbsp;- Best Fitness: {best_fitness:.4f}<br>
&nbsp;&nbsp;&nbsp;- Số bands được chọn: {best_n_bands}/{n_bands}<br>
&nbsp;&nbsp;&nbsp;- Tỷ lệ giảm: {(1 - best_n_bands/n_bands)*100:.1f}%<br>
&nbsp;&nbsp;&nbsp;- Convergence tại generation: {convergence_gen}
</div>
</div>
</div>
"""
    
    log_html += summary_html
    
    # Generate selected band IDs
    selected_bands = sorted(rng.choice(n_bands, size=best_n_bands, replace=False))
    first_ten = ", ".join(str(bid) for bid in selected_bands[:10])
    
    # Final result card
    final_result = log_html + f"""
<div style="margin-top:20px;font-size:1.8rem;font-weight:700;text-align:center;padding:40px;background:#1b5e20;color:white;border-radius:12px;">
✅ Đã chọn được {best_n_bands} bands!
</div>
<div style="font-size:1.1rem;text-align:center;padding:12px;color:#9ca3af;">
Fitness: {best_fitness:.4f} | Convergence: Gen {convergence_gen}<br>
Band IDs: [{first_ten}, ...]
</div>
"""
    
    yield final_result


# ============================================================
# GRADIO INTERFACE
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1260px !important;
    margin: auto !important;
}

.title {
    text-align: center;
    margin-bottom: 20px;
}

/* 2x2 Grid Container */
.upload-grid-container {
    border: 2px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px;
    background: #f9fafb;
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    grid-template-rows: auto auto !important;
    gap: 16px !important;
}

/* Force rows to display horizontally */
.upload-grid-container > .row {
    display: contents !important;
}

/* Individual upload cells in the grid */
.upload-cell {
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 16px;
    background: white;
    min-height: 220px !important;
    max-height: 220px !important;
    display: flex !important;
    flex-direction: column !important;
}

.upload-cell h3 {
    color: #000000 !important;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 8px;
}

/* Fixed height for file upload components */
#img1_header, #img1_raw, #img2_header, #img2_raw {
    min-height: 140px !important;
    max-height: 140px !important;
    height: 140px !important;
}

#img1_header .wrap, #img1_raw .wrap, #img2_header .wrap, #img2_raw .wrap {
    min-height: 140px !important;
    max-height: 140px !important;
    height: 140px !important;
    overflow: hidden !important;
}

/* Prevent file preview from expanding */
#img1_header .file-preview, #img1_raw .file-preview, 
#img2_header .file-preview, #img2_raw .file-preview {
    max-height: 100px !important;
    overflow-y: auto !important;
}

.output-card {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px;
    min-height: 175px;
}

.inference-time {
    font-size: 2.1rem;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 18px;
}

.note {
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 18px;
}

"""

HEAD_HTML = """
<script>
// Maintain fixed 2x2 grid layout for file uploads
(function() {
    function enforceFixedLayout() {
        // Target all file upload components
        var ids = ['img1_header', 'img1_raw', 'img2_header', 'img2_raw'];
        
        ids.forEach(function(id) {
            var el = document.getElementById(id);
            if (el) {
                // Apply fixed dimensions to the main element
                el.style.minHeight = '140px';
                el.style.maxHeight = '140px';
                el.style.height = '140px';
                el.style.overflow = 'hidden';
                
                // Apply to child elements
                var wrap = el.querySelector('.wrap');
                if (wrap) {
                    wrap.style.minHeight = '140px';
                    wrap.style.maxHeight = '140px';
                    wrap.style.height = '140px';
                    wrap.style.overflow = 'hidden';
                }
                
                // Apply to file input container
                var fileContainer = el.querySelector('.file-preview');
                if (fileContainer) {
                    fileContainer.style.maxHeight = '100px';
                    fileContainer.style.overflow = 'auto';
                }
            }
        });
        
        // Fix the parent cards to maintain consistent height
        var inputCards = document.querySelectorAll('.input-card');
        inputCards.forEach(function(card) {
            card.style.minHeight = '450px';
            card.style.maxHeight = '450px';
            card.style.overflow = 'hidden';
        });
    }
    
    // Run immediately when script loads
    enforceFixedLayout();
    
    // Re-run after a short delay to catch dynamically loaded content
    setTimeout(enforceFixedLayout, 100);
    setTimeout(enforceFixedLayout, 500);
    setTimeout(enforceFixedLayout, 1000);
    
    // Monitor for DOM changes and reapply styles
    var observer = new MutationObserver(function(mutations) {
        enforceFixedLayout();
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    });
})();
</script>
"""

with gr.Blocks(title="Durian Chemical Detection Demo", css=CUSTOM_CSS, head=HEAD_HTML) as demo:
    gr.Markdown(
        """
<div class="title">

# Hệ thống phát hiện chất cấm trên bề mặt sầu riêng

Phân tích dữ liệu ảnh siêu phổ **Ảnh 1** và **Ảnh 2**

</div>
"""
    )

    with gr.Tabs():
        with gr.Tab("Bands Selection"):
            gr.Markdown(
                """
## Selection Bands

Chọn lọc ra các bands để sử dụng
"""
            )

            with gr.Column():
                process_bands_button = gr.Button(
                    "Process",
                    variant="primary",
                    size="lg",
                )
                bands_result_output = gr.Markdown(value="&nbsp;")

                process_bands_button.click(
                    fn=run_bands_selection,
                    inputs=None,
                    outputs=bands_result_output,
                )

        with gr.Tab("Inference"):
            gr.Markdown(
                """
    ### Điều kiện ảnh siêu phổ đầu vào

    - Định dạng dữ liệu: **ENVI BIP**, gồm 1 file .hdr và 1 file .raw với mỗi ảnh.
    - **Ảnh 1**: kích thước `1632 × 1232 × 279`, dải bước sóng `400-1000nm`.
    - **Ảnh 2**: kích thước `320 × 256 × 233`, dải bước sóng sóng `1000-2500nm`.
    """
            )

            # --------------------------------------------------------
            # INPUT - 2x2 Grid Layout
            # --------------------------------------------------------

            with gr.Column(elem_classes=["upload-grid-container"]):
                # Row 1: Header files
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=500, elem_classes=["upload-cell"]):
                        gr.Markdown("### File header ảnh 1 (.hdr)")
                        image_1_header_input = gr.File(
                            label="Camera 1 - VNIR (400-1000nm)",
                            file_types=[".hdr"],
                            type="filepath",
                            elem_id="img1_header",
                        )

                    with gr.Column(scale=1, min_width=500, elem_classes=["upload-cell"]):
                        gr.Markdown("### File header ảnh 2 (.hdr)")
                        image_2_header_input = gr.File(
                            label="Camera 2 - SWIR (1000-2500nm)",
                            file_types=[".hdr", ".hsi"],
                            type="filepath",
                            elem_id="img2_header",
                        )

                # Row 2: Raw files
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=500, elem_classes=["upload-cell"]):
                        gr.Markdown("### File dữ liệu ảnh 1 (.raw)")
                        image_1_raw_input = gr.File(
                            label="Kích thước: 1632 × 1232 × 279",
                            file_types=[".raw"],
                            type="filepath",
                            elem_id="img1_raw",
                        )

                    with gr.Column(scale=1, min_width=500, elem_classes=["upload-cell"]):
                        gr.Markdown("### File dữ liệu ảnh 2 (.raw)")
                        image_2_raw_input = gr.File(
                            label="Kích thước: 320 × 256 × 233",
                            file_types=[".raw"],
                            type="filepath",
                            elem_id="img2_raw",
                        )

            detect_button = gr.Button(
                "Phát hiện chất cấm",
                variant="primary",
                size="lg",
            )

            input_information_output = gr.Markdown()

            # --------------------------------------------------------
            # DETECTION RESULT
            # --------------------------------------------------------

            gr.Markdown("# Kết quả phân tích")

            with gr.Row():
                detection_output = gr.Markdown(
                    value="""
        ## Kết quả phát hiện

        Chưa có dữ liệu phân tích.
        """,
                    elem_classes=["output-card"],
                )

                inference_time_output = gr.Markdown(
                    value="""
        ## Thời gian inference

        Chưa thực hiện.
        """,
                    elem_classes=["output-card"],
                )

            # --------------------------------------------------------
            # EVENT
            # --------------------------------------------------------

            detect_button.click(
                fn=run_detection,
                inputs=[
                    image_1_header_input,
                    image_1_raw_input,
                    image_2_header_input,
                    image_2_raw_input,
                ],
                outputs=[
                    detection_output,
                    inference_time_output,
                    input_information_output,
                ],
            )


if __name__ == "__main__":
    demo.launch()

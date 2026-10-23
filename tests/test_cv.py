import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from fpextract.cv.crop import pad_frac, refine_crop, snap_to_content  # noqa: E402
from fpextract.cv.dewatermark import remove_tiled_watermark  # noqa: E402
from fpextract.cv.normalize import normalize_background  # noqa: E402


def test_pad_frac_clamps():
    assert pad_frac([0.01, 0.5, 0.99, 0.6], 0.05) == [0.0, 0.45, 1.0, 0.65]


def test_snap_to_content_finds_drawn_box():
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 80), (420, 300), (0, 0, 0), 6)
    x0, y0, x1, y1 = snap_to_content(img, margin_frac=0.0)
    assert abs(x0 - 97) <= 4 and abs(y0 - 77) <= 4 and abs(x1 - 424) <= 4 and abs(y1 - 304) <= 4


def test_snap_on_dark_background():
    img = np.full((300, 300, 3), 20, np.uint8)
    cv2.rectangle(img, (60, 60), (220, 240), (255, 255, 255), 4)
    x0, y0, x1, y1 = snap_to_content(img, margin_frac=0.0)
    assert x0 <= 60 and y0 <= 60 and x1 >= 220 and y1 >= 240


def test_refine_crop_tightens_loose_llm_box(sample_images):
    img = Image.open(sample_images[1]).convert("RGB")  # type-a1 sheet
    W, H = img.size
    # a deliberately loose box around the drawing (walls span ~220..1100 × 180..680 px)
    loose = [150 / W, 120 / H, 1250 / W, 720 / H]
    tight = refine_crop(img, loose, pad=0.0, margin_frac=0.0)
    assert tight[0] > loose[0] and tight[1] > loose[1] and tight[2] < loose[2] and tight[3] < loose[3]
    px = [tight[0] * W, tight[1] * H, tight[2] * W, tight[3] * H]
    assert abs(px[0] - 220) < 30 and abs(px[1] - 180) < 30 and abs(px[2] - 1100) < 30 and abs(px[3] - 680) < 30


def test_normalize_background_whitens_tint():
    img = np.full((200, 200, 3), (200, 220, 240), np.uint8)
    cv2.line(img, (20, 100), (180, 100), (0, 0, 0), 3)
    out = normalize_background(img)
    assert out[10, 10].min() >= 250            # background → white
    assert out[100, 100].max() < 100           # line preserved dark


def test_tiled_watermark_passthrough_on_clean_image():
    img = np.full((600, 800, 3), 250, np.uint8)
    clean, info = remove_tiled_watermark(img)
    assert info["matched"] is False and info["skipped"]
    assert np.array_equal(clean, img)


def test_tiled_watermark_removes_repeated_text():
    img = np.full((900, 1400, 3), 245, np.uint8)
    for y in range(40, 900, 150):
        for x in range(30, 1400, 300):
            cv2.putText(img, "SAMPLE", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (225, 225, 225), 3)
    clean, info = remove_tiled_watermark(img, template_width=(100, 400))
    assert info["matched"] is True and info["n_detected"] >= 5
    assert int((clean < 240).sum()) < int((img < 240).sum())

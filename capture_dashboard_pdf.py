import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


URL = "https://rundown-pranker-upswing.ngrok-free.dev"
OUT_DIR = Path("outputs/pitch_assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAGES: List[Tuple[str, List[str]]] = [
    ("overview", ["Обзор сегментов", "Overview Segments"]),
    ("eda", ["Анализ данных (EDA)", "Data Analysis (EDA)"]),
    ("shap", ["SHAP — факторы оттока", "SHAP — churn factors"]),
    ("top_users", ["Топ пользователи", "Top users"]),
]


def capture_screenshots() -> List[Path]:
    screenshots: List[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1920, "height": 1080},
            extra_http_headers={"ngrok-skip-browser-warning": "1"},
        )
        page.goto(URL, wait_until="networkidle", timeout=120000)
        page.wait_for_timeout(2500)

        # Fallback: if ngrok warning still appears, click "Visit Site".
        try:
            if page.locator("text=Visit Site").first.is_visible(timeout=1500):
                page.locator("text=Visit Site").first.click(timeout=5000)
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(2500)
        except Exception:
            pass

        # Remove occasional cookie/popup overlays if present.
        for selector in ["button:has-text('Accept')", "button:has-text('OK')"]:
            try:
                if page.locator(selector).first.is_visible(timeout=1000):
                    page.locator(selector).first.click()
            except Exception:
                pass

        # Ensure Streamlit page is loaded (not ngrok interstitial).
        try:
            page.locator("text=Freedom Churn Intelligence").first.wait_for(timeout=20000)
        except Exception:
            print("Warning: dashboard title not detected, but continuing capture.")

        # Always capture initial full page (most important for pitch).
        shot_path = OUT_DIR / "overview_full.png"
        page.screenshot(path=str(shot_path), full_page=True)
        screenshots.append(shot_path)

        # Also capture viewport-based slides to highlight details quickly.
        for idx, y in enumerate([0, 700, 1500], start=1):
            page.evaluate(f"window.scrollTo(0, {y});")
            page.wait_for_timeout(1200)
            viewport_path = OUT_DIR / f"overview_view_{idx}.png"
            page.screenshot(path=str(viewport_path), full_page=False)
            screenshots.append(viewport_path)

        # Try to navigate sidebar radio pages (non-fatal if labels are unavailable).
        for key, labels in PAGES:
            clicked = False
            for label in labels:
                try:
                    page.locator(f"label:has-text('{label}')").first.click(timeout=5000)
                    clicked = True
                    break
                except Exception:
                    try:
                        page.locator(f"text={label}").first.click(timeout=3000)
                        clicked = True
                        break
                    except Exception:
                        continue

            if clicked:
                page.wait_for_timeout(2000)
                page.evaluate("window.scrollTo(0, 0);")
                page.wait_for_timeout(600)
                tab_path = OUT_DIR / f"{key}.png"
                page.screenshot(path=str(tab_path), full_page=True)
                screenshots.append(tab_path)
            else:
                print(f"Warning: could not open sidebar page '{key}'.")

        browser.close()

    return screenshots


def images_to_pdf(images: List[Path], output_pdf: Path) -> None:
    pil_images = []
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        pil_images.append(img)

    if not pil_images:
        raise RuntimeError("No images captured; cannot create PDF.")

    first, rest = pil_images[0], pil_images[1:]
    first.save(output_pdf, save_all=True, append_images=rest)


def main():
    print("Capturing dashboard screenshots...")
    images = capture_screenshots()
    print(f"Captured {len(images)} screenshots.")

    pdf_path = OUT_DIR / "dashboard_screenshots_for_pitch.pdf"
    images_to_pdf(images, pdf_path)
    print(f"Saved PDF: {pdf_path.resolve()}")


if __name__ == "__main__":
    main()


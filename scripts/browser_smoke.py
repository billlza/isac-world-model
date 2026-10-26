"""Execute the generated offline viewer in a real browser; fail on script/network errors."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from group_demo import CASES, prepare
from playwright.sync_api import expect, sync_playwright


def check(directory: Path, *, executable: str | None = None) -> dict:
    errors, requests = [], []
    with sync_playwright() as manager:
        browser = manager.chromium.launch(**({"executable_path": executable} if executable else {}))
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("pageerror", lambda error: errors.append(str(error)))

            def block(route):
                requests.append(route.request.url)
                route.abort()

            page.route("**/*", block)
            page.set_content((directory / "index.html").read_text(encoding="utf-8"))
            expect(page.locator("h1")).to_contain_text("Asynchronous")
            for name, _, _ in CASES:
                expect(page.locator(f'a[href="{name}/replay.html"]')).to_have_count(1)
            for name, _, _ in CASES:
                page.close()
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.route("**/*", block)
                page.set_content((directory / name / "replay.html").read_text(encoding="utf-8"))
                expect(page.locator("#run option")).to_have_count(4)
                expect(page.locator("#clock")).to_contain_text("0.00 s")
                page.locator("#run").select_option(index=3)
                page.locator("#slider").evaluate("el => {el.value=5;el.dispatchEvent(new Event('input'))}")
                expect(page.locator("#clock")).to_contain_text("0.50 s")
                page.locator("#reset").click()
                expect(page.locator("#clock")).to_contain_text("0.00 s")
                page.locator("#play").click()
                expect(page.locator("#clock")).not_to_contain_text("0.00 s")
                page.locator("#play").click()
                page.set_viewport_size({"width": 390, "height": 844})
                expect(page.locator("#scene")).to_be_visible()
                expect(page.locator("#limitations")).to_contain_text("synthetic")
            if errors or requests:
                raise AssertionError(f"Browser errors={errors}; external requests={requests}")
            return {
                "status": "passed",
                "cases": len(CASES),
                "page_errors": errors,
                "external_requests": requests,
            }
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--executable", help="Explicit local browser override; omitted in pinned CI")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="isac-browser-") as folder:
            demo = Path(folder) / "demo"
            prepare(demo, quick=True)
            result = check(demo, executable=args.executable)
    except Exception as exc:
        args.out.write_text(
            json.dumps({"status": "failed", "error": str(exc)}, indent=2) + "\n", encoding="utf-8"
        )
        raise
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

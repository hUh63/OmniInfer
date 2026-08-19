import re
import unittest
import urllib.parse
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPOSITORY_ROOT / "README.md"


class RootReadmeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = README_PATH.read_text(encoding="utf-8")

    def test_product_sections_are_concise_and_ordered(self):
        sections = (
            "## Quick Start",
            "## Demo",
            "## News",
            "## About",
            "## Platform Support",
            "## Documentation",
            "## Architecture",
            "## Contributing",
            "## Citation",
            "## License",
        )
        offsets = [self.readme.index(section) for section in sections]
        self.assertEqual(offsets, sorted(offsets))

    def test_primary_entry_points_and_status_badges_are_visible(self):
        for marker in (
            'href="#quick-start"',
            'href="#documentation"',
            "https://github.com/omnimind-ai/OmniInfer/releases",
            "actions/workflows/main-platform-ci.yml/badge.svg",
            "img.shields.io/github/v/release/omnimind-ai/OmniInfer",
            "img.shields.io/github/license/omnimind-ai/OmniInfer",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.readme)

    def test_three_release_installers_and_first_run_are_copyable(self):
        for platform in ("Linux x64", "macOS arm64", "Windows x64 PowerShell"):
            with self.subTest(platform=platform):
                self.assertIn(f"<th>{platform}</th>", self.readme)
        self.assertGreaterEqual(self.readme.count("scripts/install.sh | bash"), 2)
        self.assertIn("scripts/install.ps1 | iex", self.readme)
        self.assertIn("1. Run `omniinfer` in a terminal.", self.readme)
        self.assertIn("2. Choose a compatible backend.", self.readme)
        self.assertIn("3. Select a local model and start chatting.", self.readme)

    def test_implementation_details_stay_in_subdocuments(self):
        for detail in (
            "install-from-source.sh",
            "install-from-source.ps1",
            "--state-root",
            "--runtime-root",
            "cloudflared",
            "vllm-wsl2-cuda",
            "vllm-wsl2-rocm",
        ):
            with self.subTest(detail=detail):
                self.assertNotIn(detail, self.readme)

        cli = (REPOSITORY_ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
        remote = (REPOSITORY_ROOT / "docs" / "remote-access.md").read_text(
            encoding="utf-8"
        )
        installation = (
            REPOSITORY_ROOT / "docs" / "installation.md"
        ).read_text(encoding="utf-8")
        build = (REPOSITORY_ROOT / "docs" / "build.md").read_text(encoding="utf-8")
        self.assertIn("#### Desktop application integration", cli)
        self.assertIn("#### Windows vLLM through WSL2", cli)
        self.assertIn("## Managed cloudflared", remote)
        self.assertIn("## Complete Source Setup", installation)
        self.assertIn("## Manual Installation", installation)
        self.assertIn("## Remove the Release CLI", installation)
        self.assertIn("## Windows", build)
        self.assertIn("## Linux", build)
        self.assertIn("## macOS", build)

    def test_news_copy_is_preserved(self):
        self.assertIn(
            "- **2026-08-14** — 🚀 **Day-0 support for Qwen3.8-27B.** "
            "OmniInfer is ready for Qwen's latest 27B vision-language model "
            "from day one.",
            self.readme,
        )

    def test_local_readme_links_resolve(self):
        markdown_targets = re.findall(r"!?(?:\[[^\]]*\])\(([^)]+)\)", self.readme)
        html_targets = re.findall(r"(?:href|src)=\"([^\"]+)\"", self.readme)
        missing = []
        for raw_target in markdown_targets + html_targets:
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            parsed = urllib.parse.urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            relative_path = urllib.parse.unquote(parsed.path)
            if relative_path and not (REPOSITORY_ROOT / relative_path).exists():
                missing.append(target)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_demo_assets_notebooks_and_results_use_organized_directories(self):
        self.assertTrue((ROOT / "data" / "examples").is_dir())
        self.assertTrue((ROOT / "notebooks" / "training").is_dir())
        self.assertTrue((ROOT / "notebooks" / "inference" / "Use_Model.ipynb").is_file())
        self.assertTrue((ROOT / "evaluation" / "results").is_dir())

    def test_evaluation_results_are_tracked(self):
        expected_results = {
            "DINOv2_results.json",
            "EffoVPR_G_results.json",
            "EffoVPR_R_results.json",
            "EffoVPR_ZS_results.json",
        }
        tracked_results = subprocess.run(
            ["git", "ls-files", "evaluation/results"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertEqual(
            {Path(path).name for path in tracked_results},
            expected_results,
        )

    def test_streamlit_app_reads_demo_images_from_data_examples(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('ROOT / "data" / "examples"', app_source)


if __name__ == "__main__":
    unittest.main()

"""End-to-End Integration Test for NovaScientist Pipeline."""

import os
import shutil
import tempfile
import zipfile
import pytest
from cli import run_pipeline


@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        topic = "Low-Compute Dynamic Graph Representation under Quantized Memory"
        await run_pipeline(topic=topic, num_seeds=3, output_dir=tmpdir)

        # Check that dist contains the zip file
        zip_files = [f for f in os.listdir(tmpdir) if f.endswith(".zip")]
        assert len(zip_files) == 1
        zip_path = os.path.join(tmpdir, zip_files[0])
        assert os.path.exists(zip_path)

        # Inspect ZIP contents
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            assert "main.tex" in names
            assert "references.bib" in names
            assert "IEEEtran.cls" in names
            assert "artifacts/metrics.json" in names
            assert "README.md" in names
            assert any(n.startswith("figures/") for n in names)

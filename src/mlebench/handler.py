"""
Entropic Atlas — MLE-Bench Domain Handler

Orchestrates the complete ML competition pipeline:
1. Extract competition data from tar.gz
2. Analyze competition description
3. Generate ML pipeline code
4. Execute code to produce submission.csv
5. Self-heal on failure (retry with error context)
6. Return submission CSV bytes

Receives: instructions text + competition.tar.gz from green agent
Returns: (csv_bytes, summary_text)
"""

import base64
import io
import logging
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

from config import Config
from llm import LLMClient
from mlebench.analyzer import CompetitionAnalyzer
from mlebench.codegen import MLCodeGenerator
from mlebench.executor import CodeExecutor

logger = logging.getLogger("entropic-atlas.mlebench")


class MLEBenchHandler:
    """Handle MLE-Bench competition tasks end-to-end."""

    def __init__(self, config: Config, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.analyzer = CompetitionAnalyzer(llm)
        self.codegen = MLCodeGenerator(llm)
        self.executor = CodeExecutor(timeout=config.code_execution_timeout)

    async def handle(
        self,
        text: str,
        file_parts: list[tuple[str, str, str | bytes]],
        updater: TaskUpdater,
    ) -> tuple[bytes, str]:
        """
        Process an MLE-Bench competition end-to-end.

        Args:
            text: Instructions text from green agent
            file_parts: List of (name, mime_type, data) — should include competition.tar.gz
            updater: A2A task updater for progress reporting

        Returns:
            (csv_bytes, summary_text) tuple
        """
        # 1. Extract competition tar.gz to temp directory
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Extracting competition data..."),
        )

        work_dir = self._extract_competition(file_parts)
        data_dir = self._find_data_dir(work_dir)
        logger.info(f"Competition data extracted to {work_dir}, data at {data_dir}")

        # 2. Read competition description
        description = self._read_description(data_dir)
        file_listing = self._list_data_files(data_dir)
        data_preview = self._preview_data(data_dir)

        logger.info(f"Description: {len(description)} chars, files: {file_listing[:200]}")

        # 3. Analyze competition
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Analyzing competition..."),
        )

        analysis = await self.analyzer.analyze(
            description=description,
            file_listing=file_listing,
            data_preview=data_preview,
        )

        logger.info(
            f"Analysis: type={analysis.task_type}, metric={analysis.metric}, "
            f"strategy={analysis.strategy}"
        )

        # 4. Generate ML pipeline code
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Generating {analysis.strategy} ML pipeline..."
            ),
        )

        submission_path = str(work_dir / "submission.csv")
        code = await self.codegen.generate(
            description=description,
            data_dir=str(data_dir),
            file_listing=file_listing,
            data_preview=data_preview,
            analysis=analysis,
            submission_path=submission_path,
        )

        # 5. Execute code with self-healing retries
        csv_bytes = None
        for attempt in range(self.config.max_code_iterations):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Running ML pipeline (attempt {attempt + 1}/{self.config.max_code_iterations})..."
                ),
            )

            csv_bytes = await self.executor.execute(
                code=code,
                working_dir=work_dir,
                submission_path=Path(submission_path),
            )

            if csv_bytes is not None:
                logger.info(f"Pipeline succeeded on attempt {attempt + 1}")
                break

            # Self-heal: fix the code based on the error
            if attempt < self.config.max_code_iterations - 1:
                logger.info(
                    f"Pipeline failed on attempt {attempt + 1}, self-healing..."
                )
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Pipeline failed, fixing code (attempt {attempt + 2})..."
                    ),
                )

                code = await self.codegen.fix(
                    code=code,
                    error=self.executor.last_error or "Unknown error",
                    stdout=self.executor.last_stdout,
                    description=description,
                    file_listing=file_listing,
                )

        if csv_bytes is None:
            # Last resort: generate a dummy submission
            logger.error("All attempts failed, generating dummy submission")
            csv_bytes = self._generate_dummy_submission(data_dir, analysis)

        summary = (
            f"Competition: {analysis.task_type} ({analysis.metric})\n"
            f"Strategy: {analysis.strategy}\n"
            f"Submission: {len(csv_bytes)} bytes"
        )
        return csv_bytes, summary

    def _extract_competition(
        self, file_parts: list[tuple[str, str, str | bytes]]
    ) -> Path:
        """Extract competition.tar.gz to a temporary directory."""
        tar_data = None
        for name, mime, data in file_parts:
            if name and ("tar" in name or "gz" in name):
                tar_data = data
                break
            if mime and ("tar" in mime or "gzip" in mime):
                tar_data = data
                break

        if tar_data is None:
            # Use the first file attachment as tar
            if file_parts:
                _, _, tar_data = file_parts[0]
            else:
                raise ValueError("No competition data file received")

        # Decode if base64
        if isinstance(tar_data, str):
            if tar_data.startswith("data:"):
                tar_data = tar_data.split(",", 1)[1]
            tar_data = base64.b64decode(tar_data)

        # Extract to temp directory
        work_dir = Path(tempfile.mkdtemp(prefix="atlas_mle_"))
        try:
            with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as tar:
                tar.extractall(work_dir, filter="data")
        except tarfile.ReadError:
            # Try uncompressed tar
            with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:") as tar:
                tar.extractall(work_dir, filter="data")

        logger.info(f"Extracted competition to {work_dir}")
        return work_dir

    def _find_data_dir(self, work_dir: Path) -> Path:
        """Find the data directory within extracted competition."""
        # MLE-Bench structure: home/data/
        candidates = [
            work_dir / "home" / "data",
            work_dir / "data",
            work_dir,
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        return work_dir

    def _read_description(self, data_dir: Path) -> str:
        """Read competition description."""
        for name in ["description.md", "README.md", "description.txt"]:
            desc_path = data_dir / name
            if desc_path.exists():
                return desc_path.read_text(errors="replace")

        # Look one level up
        parent = data_dir.parent
        for name in ["description.md", "README.md"]:
            desc_path = parent / name
            if desc_path.exists():
                return desc_path.read_text(errors="replace")

        return "[No description file found]"

    def _list_data_files(self, data_dir: Path) -> str:
        """List all data files with sizes."""
        lines = []
        for f in sorted(data_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(data_dir)
                size = f.stat().st_size
                if size > 1_000_000:
                    size_str = f"{size / 1_000_000:.1f}MB"
                elif size > 1_000:
                    size_str = f"{size / 1_000:.1f}KB"
                else:
                    size_str = f"{size}B"
                lines.append(f"  - {rel} ({size_str})")
        return "\n".join(lines) if lines else "  [No files found]"

    def _preview_data(self, data_dir: Path, max_rows: int = 5) -> str:
        """Preview CSV files in the data directory."""
        previews = []
        csv_files = sorted(data_dir.glob("*.csv"))[:3]  # first 3 CSVs

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, nrows=max_rows)
                previews.append(
                    f"### {csv_path.name}\n"
                    f"Shape: {df.shape}\n"
                    f"Columns: {list(df.columns)}\n"
                    f"Dtypes:\n{df.dtypes.to_string()}\n"
                    f"Head:\n{df.to_string()}\n"
                )
            except Exception as e:
                previews.append(f"### {csv_path.name}\nError reading: {e}\n")

        return "\n".join(previews) if previews else "[No CSV files to preview]"

    def _generate_dummy_submission(
        self, data_dir: Path, analysis
    ) -> bytes:
        """Generate a minimal valid submission as last resort."""
        # Try to read test.csv and create a dummy submission
        test_path = data_dir / "test.csv"
        if not test_path.exists():
            for candidate in data_dir.glob("test*.csv"):
                test_path = candidate
                break

        try:
            test_df = pd.read_csv(test_path)
            submission = pd.DataFrame()

            # Try to find ID column
            for col in test_df.columns:
                if "id" in col.lower():
                    submission[col] = test_df[col]
                    break

            # Add target column with dummy value
            if analysis.target_column:
                submission[analysis.target_column] = 0
            else:
                submission["target"] = 0

            buf = io.BytesIO()
            submission.to_csv(buf, index=False)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Dummy submission failed: {e}")
            return b"id,target\n0,0\n"

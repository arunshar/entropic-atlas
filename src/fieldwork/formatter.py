"""
Entropic Atlas — Output Format Matcher

Ensures answers match the expected output format for FieldWorkArena scoring.
Critical because the green agent uses exact_match, must_include, json_match, etc.
A correct answer in the wrong format scores 0.
"""

import json
import logging
import re

logger = logging.getLogger("entropic-atlas.fieldwork.formatter")


class AnswerFormatter:
    """Format raw LLM answers to match expected output formats."""

    def format_answer(self, raw_answer: str, output_format: str) -> str:
        """
        Ensure answer matches the expected output format.

        Args:
            raw_answer: The raw answer from the reasoner
            output_format: The expected format description from the green agent

        Returns:
            Formatted answer string
        """
        fmt_lower = output_format.lower()
        answer = raw_answer.strip()

        # JSON format
        if "json" in fmt_lower or output_format.strip().startswith("{"):
            return self._format_json(answer)

        # Numeric format
        if any(kw in fmt_lower for kw in ["number", "count", "integer", "how many"]):
            return self._format_numeric(answer)

        # Boolean format
        if "yes/no" in fmt_lower or "yes or no" in fmt_lower:
            return self._format_boolean(answer)

        # List format
        if "list" in fmt_lower or "comma" in fmt_lower:
            return self._format_list(answer)

        # Strip markdown artifacts
        answer = self._strip_markdown(answer)

        return answer

    def _format_json(self, answer: str) -> str:
        """Extract and clean JSON from answer."""
        # Try to parse the whole answer as JSON first
        try:
            parsed = json.loads(answer)
            return json.dumps(parsed)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object from answer
        json_match = re.search(r'\{.*\}', answer, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return json.dumps(parsed)
            except json.JSONDecodeError:
                pass

        # Try JSON array
        json_match = re.search(r'\[.*\]', answer, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return json.dumps(parsed)
            except json.JSONDecodeError:
                pass

        # Return as-is if no JSON found
        logger.warning(f"Could not extract JSON from answer: {answer[:200]}")
        return answer

    def _format_numeric(self, answer: str) -> str:
        """Extract numeric value from answer."""
        # Try whole answer as number
        try:
            val = float(answer)
            return str(int(val)) if val == int(val) else str(val)
        except ValueError:
            pass

        # Extract first number from text
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            val = float(numbers[0])
            return str(int(val)) if val == int(val) else str(val)

        return answer

    def _format_boolean(self, answer: str) -> str:
        """Normalize to yes/no."""
        answer_lower = answer.lower().strip()
        if answer_lower in ("yes", "true", "correct", "affirmative"):
            return "yes"
        elif answer_lower in ("no", "false", "incorrect", "negative"):
            return "no"

        # Check if answer starts with yes/no
        if answer_lower.startswith("yes"):
            return "yes"
        elif answer_lower.startswith("no"):
            return "no"

        return answer

    def _format_list(self, answer: str) -> str:
        """Clean up list formatting."""
        # Remove bullet points and numbering
        lines = answer.strip().split("\n")
        items = []
        for line in lines:
            cleaned = re.sub(r'^\s*[-*\d.)\]]+\s*', '', line).strip()
            if cleaned:
                items.append(cleaned)
        return ", ".join(items) if items else answer

    def _strip_markdown(self, answer: str) -> str:
        """Remove markdown formatting artifacts."""
        # Remove code blocks
        answer = re.sub(r'```[\s\S]*?```', lambda m: m.group().strip('`').strip(), answer)
        # Remove inline code
        answer = re.sub(r'`([^`]+)`', r'\1', answer)
        # Remove bold/italic
        answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
        answer = re.sub(r'\*([^*]+)\*', r'\1', answer)
        return answer.strip()

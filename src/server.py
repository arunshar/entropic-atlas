"""
Entropic Atlas — A2A Purple Agent Server

Spatial-aware research agent for AgentX-AgentBeats Phase 2 Sprint 2.
Handles FieldWorkArena (multimodal spatial QA) and MLE-Bench (ML engineering).
"""

import argparse
import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from executor import Executor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("entropic-atlas")


def main():
    parser = argparse.ArgumentParser(description="Run Entropic Atlas purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skills = [
        AgentSkill(
            id="fieldwork-research",
            name="Multimodal Field Research",
            description=(
                "Analyzes factory, warehouse, and retail environments from images, "
                "videos, PDFs, and documents. Spatial reasoning with structured scene "
                "graphs, safety inspection, and formatted reporting."
            ),
            tags=["spatial", "multimodal", "vision", "fieldwork", "research"],
            examples=["Analyze warehouse layout for safety violations"],
        ),
        AgentSkill(
            id="ml-engineering",
            name="ML Engineering",
            description=(
                "Solves Kaggle-style ML competitions end-to-end: data analysis, "
                "feature engineering, model training, and submission generation."
            ),
            tags=["ml", "kaggle", "data-science", "code-generation"],
            examples=["Train a model for the spaceship-titanic competition"],
        ),
    ]

    agent_card = AgentCard(
        name="Entropic Atlas",
        description=(
            "Spatial-aware research agent with multimodal perception and ML engineering. "
            "Combines entropy-guided reasoning with structured spatial scene graphs "
            "for field work analysis, and systematic ML pipelines for competition solving. "
            "Built for AgentX-AgentBeats Phase 2 Sprint 2 Research Agent track."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print("=" * 60)
    print("Entropic Atlas — Purple Agent")
    print("=" * 60)
    print(f"Server: http://{args.host}:{args.port}/")
    print(f"Agent Card: {agent_card.url}")
    print()
    print("Skills:")
    for skill in skills:
        print(f"  - {skill.name}: {skill.description[:80]}...")
    print("=" * 60)

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()

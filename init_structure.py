from pathlib import Path
import json

structure = {
    "agent": ["__init__.py", "graph.py", "nodes.py", "prompts.py"],
    "checklist": ["__init__.py", "schema.py", "sample_checklist.json"],
    "data": [],
    "docs": [],
    "evaluation": ["__init__.py", "evaluator.py", "report_generator.py"],
}

project_root = Path.cwd()

for folder, files in structure.items():
    folder_path = project_root / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    for file in files:
        (folder_path / file).touch()

print("✅ Folder structure created.")

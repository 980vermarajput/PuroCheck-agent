from typing import List, Optional
from pydantic import BaseModel

# Each item inside a section
class ChecklistItem(BaseModel):
    requirement: str
    puroRequires: Optional[str] = None
    documentsNeeded: Optional[str] = None
    puroWillCheckFor: Optional[str] = None
    puroLooksFor: Optional[str] = None
    parameter: Optional[str] = None
    notes: Optional[str] = None
    puroWillValidate: Optional[str] = None
    status: bool = False

# Each section in the checklist
class ChecklistSection(BaseModel):
    title: str
    note: Optional[str] = None
    items: List[ChecklistItem]

# Root checklist structure
class FullChecklist(BaseModel):
    sections: List[ChecklistSection]

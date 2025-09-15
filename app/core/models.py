from pydantic import BaseModel
from typing import Optional, Union
from datetime import date, datetime

class Task(BaseModel):
    task_title: str
    description: Optional[str] = None
    #due_date: Optional[Union[date, datetime]] = None
    due_date: Optional[str] = None
    requestor: Optional[str] = None
    completion_status: int # 0 for incomplete, 1 for complete
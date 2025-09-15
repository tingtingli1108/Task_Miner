"""
Prompt Management System for TaskMiner RAG Pipeline

This module provides a flexible system for managing and evaluating different prompt variants
for task detection and extraction.
"""

import yaml
import json
import os
from typing import Dict, Any, List, Optional
from langchain.prompts import ChatPromptTemplate
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PromptType(Enum):
    TASK_DETECTION = "task_detection"
    TASK_EXTRACTION = "task_extraction"
    RETRIEVAL_FILTER = "retrieval_filter"

@dataclass
class PromptVariant:
    """Represents a specific prompt variant with metadata"""
    name: str
    version: str
    prompt_type: PromptType
    system_message: str
    user_message: str
    description: str
    expected_improvements: List[str]
    author: str = "unknown"
    created_date: str = ""

class PromptManager:
    """Manages prompt variants and enables systematic evaluation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.variants: Dict[str, Dict[str, PromptVariant]] = {
            PromptType.TASK_DETECTION.value: {},
            PromptType.TASK_EXTRACTION.value: {},
            PromptType.RETRIEVAL_FILTER.value: {}
        }
        
        # Try to load from the standard comprehensive prompts file first
        if config_path:
            self.load_from_config(config_path)
        elif os.path.exists("prompts/prompts.yaml"):
            logger.info("Loading prompts from prompts/prompts.yaml file")
            self.load_from_config("prompts/prompts.yaml")
        # check for prompts.yaml in the current directory
        elif os.path.exists("prompts.yaml"):
            logger.info("Loading prompts from prompts.yaml file in current directory")
            self.load_from_config("prompts.yaml")
        else:
            logger.info("No prompts.yaml found, loading built-in defaults")
            self._load_default_prompts()
    
    def _load_default_prompts(self):
        """Load the current production prompts as baseline variants"""
        
        # Current production task detection prompt (v1.0)
        task_detection_v1 = PromptVariant(
            name="production_baseline",
            version="v1.0",
            prompt_type=PromptType.TASK_DETECTION,
            system_message="You are an expert assistant. Your goal is to determine if the following email contains a task for {user_name} (email: {user_id}). An task is defined as anyting that the user needs to reply or take action on. It can be a request, a reminder, or any actionable item that requires the user's attention. Respond with only the word 'TRUE' or 'FALSE'.",
            user_message="Based on the following email, does it contain any actionable tasks assigned to {user_name} (email: {user_id})?\n\n\n\n**Special Rule**: If the sender's email address in the 'From:' field is the same as the user's email ({user_id}), treat the content as a personal reminder and consider it a task.\n\nEmail:\n{email_contents}",
            description="Current production prompt for task detection",
            expected_improvements=["Baseline performance"],
            author="TaskMiner Team"
        )
        
        # Current production task extraction prompt (v1.0)
        task_extraction_v1 = PromptVariant(
            name="production_baseline",
            version="v1.0",
            prompt_type=PromptType.TASK_EXTRACTION,
            system_message="You are an expert assistant that extracts tasks from email threads for a specific user. Always return the output as a single, clean JSON array of task objects.",
            user_message="""
    Extract all actionable tasks for user '{user_name}' (email: {user_id}) from the "Primary Email" below. Use the "Related Conversation History" for additional context.
    
    Each task has the following fields:
    {{
        "task_title": string,
        "description": string,
        "due_date": date,
        "requestor": string,
        "completion_status": integer (1 if the task is mentioned as complete, 0 if it is not)
    }}

    Instructions:
    - The task owner must be '{user_name}' or '{user_id}'.
    - The 'requestor' is the person who asked for the task to be done.
    - Always return the output as a JSON array (list) of tasks.
    - **IMPORTANT**: The 'due_date' must be in "YYYY-MM-DD" format. If no specific date is found, return null.
    - **Special Rule**: If the sender's email address in the 'From:' field is the same as the user's email ({user_id}), treat the content as a personal reminder. In this case, the 'requestor' and 'task_owner' are both '{user_name}'.

    ---
    ### Related Conversation History:
    {retrieved_context}
    ---
    ### Primary Email to Analyze:
    {email_contents}
    ---
    ### JSON Output:
    """,
            description="Current production prompt for task extraction",
            expected_improvements=["Baseline performance"],
            author="TaskMiner Team"
        )
        
        self.register_variant(task_detection_v1)
        self.register_variant(task_extraction_v1)
        
        # Add some experimental variants
        self._add_experimental_variants()
    
    def _add_experimental_variants(self):
        """Add experimental prompt variants for testing"""
        
        # Enhanced task detection with explicit examples
        task_detection_v2 = PromptVariant(
            name="enhanced_with_examples",
            version="v2.0",
            prompt_type=PromptType.TASK_DETECTION,
            system_message="""You are an expert assistant specializing in task identification from emails. 

A task is ANY actionable item that requires the recipient to:
- Reply to someone
- Complete an action 
- Attend a meeting
- Provide information
- Make a decision
- Follow up on something

Examples of TASKS:
- "Can you send me the report by Friday?"
- "Please review the attached document"
- "Let's schedule a meeting to discuss this"
- "Reminder: submit your timesheet"

Examples of NOT tasks:
- "FYI - the meeting was cancelled"
- "Thanks for your help yesterday"
- "Here's the information you requested"

Respond with only 'TRUE' or 'FALSE'.""",
            user_message="Does this email contain any actionable tasks for {user_name} (email: {user_id})?\n\n**Special Rule**: If the sender is {user_id}, treat as a personal reminder/task.\n\nEmail:\n{email_contents}",
            description="Enhanced prompt with explicit examples and clearer task definition",
            expected_improvements=["Better precision", "Reduced false positives", "Clearer task boundaries"],
            author="Evaluation Team"
        )
        
        # More concise task extraction
        task_extraction_v2 = PromptVariant(
            name="concise_extraction",
            version="v2.0", 
            prompt_type=PromptType.TASK_EXTRACTION,
            system_message="Extract actionable tasks from emails. Return valid JSON array only.",
            user_message="""Extract tasks for {user_name} ({user_id}) from the email below.

JSON Format:
[{{"task_title": "string", "description": "string", "due_date": "YYYY-MM-DD or null", "requestor": "string", "completion_status": 0 or 1}}]

Context: {retrieved_context}

Email: {email_contents}

JSON:""",
            description="More concise prompt focusing on essential information",
            expected_improvements=["Faster processing", "Less verbose", "More focused output"],
            author="Evaluation Team"
        )
        
        self.register_variant(task_detection_v2)
        self.register_variant(task_extraction_v2)
    
    def register_variant(self, variant: PromptVariant):
        """Register a new prompt variant"""
        prompt_type = variant.prompt_type.value
        self.variants[prompt_type][variant.name] = variant
        logger.info(f"Registered prompt variant: {variant.name} ({variant.version}) for {prompt_type}")
    
    def get_prompt_template(self, prompt_type: PromptType, variant_name: str = "production_baseline") -> ChatPromptTemplate:
        """Get a ChatPromptTemplate for the specified variant"""
        variant = self.variants[prompt_type.value].get(variant_name)
        if not variant:
            raise ValueError(f"Prompt variant '{variant_name}' not found for type {prompt_type.value}")
        
        return ChatPromptTemplate.from_messages([
            ("system", variant.system_message),
            ("user", variant.user_message)
        ])
    
    def list_variants(self, prompt_type: Optional[PromptType] = None) -> Dict[str, List[str]]:
        """List all available prompt variants"""
        if prompt_type:
            return {prompt_type.value: list(self.variants[prompt_type.value].keys())}
        
        return {
            ptype: list(variants.keys()) 
            for ptype, variants in self.variants.items()
        }
    
    def get_variant_info(self, prompt_type: PromptType, variant_name: str) -> PromptVariant:
        """Get detailed information about a specific variant"""
        variant = self.variants[prompt_type.value].get(variant_name)
        if not variant:
            raise ValueError(f"Prompt variant '{variant_name}' not found for type {prompt_type.value}")
        return variant
    
    def save_to_config(self, config_path: str):
        """Save all variants to a YAML configuration file"""
        config_data = {}
        
        for prompt_type, variants in self.variants.items():
            config_data[prompt_type] = {}
            for variant_name, variant in variants.items():
                config_data[prompt_type][variant_name] = {
                    'version': variant.version,
                    'system_message': variant.system_message,
                    'user_message': variant.user_message,
                    'description': variant.description,
                    'expected_improvements': variant.expected_improvements,
                    'author': variant.author,
                    'created_date': variant.created_date
                }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Saved prompt configuration to {config_path}")
    
    def load_from_config(self, config_path: str):
        """Load variants from a YAML configuration file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            for prompt_type_str, variants_data in config_data.items():
                prompt_type = PromptType(prompt_type_str)
                
                for variant_name, variant_data in variants_data.items():
                    variant = PromptVariant(
                        name=variant_name,
                        version=variant_data.get('version', 'unknown'),
                        prompt_type=prompt_type,
                        system_message=variant_data['system_message'],
                        user_message=variant_data['user_message'],
                        description=variant_data.get('description', ''),
                        expected_improvements=variant_data.get('expected_improvements', []),
                        author=variant_data.get('author', 'unknown'),
                        created_date=variant_data.get('created_date', '')
                    )
                    self.register_variant(variant)
            
            logger.info(f"Loaded prompt configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load prompt configuration: {e}")
            self._load_default_prompts()  # Fallback to defaults


# Global prompt manager instance
prompt_manager = PromptManager()


def get_task_detection_prompt(variant_name: str = "production_baseline") -> ChatPromptTemplate:
    """Convenience function to get task detection prompt"""
    return prompt_manager.get_prompt_template(PromptType.TASK_DETECTION, variant_name)


def get_task_extraction_prompt(variant_name: str = "production_baseline") -> ChatPromptTemplate:
    """Convenience function to get task extraction prompt"""
    return prompt_manager.get_prompt_template(PromptType.TASK_EXTRACTION, variant_name)


def get_retrieval_context_filter_prompt(variant_name: str = "production_baseline") -> ChatPromptTemplate:
    """Convenience function to get retrieval context filter prompt"""
    return prompt_manager.get_prompt_template(PromptType.RETRIEVAL_FILTER, variant_name) 
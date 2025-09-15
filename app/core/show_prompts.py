#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from core.prompt_manager import prompt_manager

print("=== AVAILABLE PROMPT VARIANTS ===")
variants = prompt_manager.list_variants()
for prompt_type, variant_list in variants.items():
    print(f"\n{prompt_type.upper()}:")
    for variant in variant_list:
        print(f"  - {variant}")

print(f"\n=== LOADED FROM: ===")
print(f"Working directory: {os.getcwd()}")
print(f"prompts/prompts.yaml exists: {os.path.exists('prompts/prompts.yaml')}") 
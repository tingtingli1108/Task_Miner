#!/usr/bin/env python3
"""
Create Comprehensive Prompts Configuration

This script creates a single YAML file containing ALL prompt variants
(baseline, experimental, and any custom ones) for the TaskMiner system.

This serves as the single source of truth for all prompts.
"""

import os
import sys
from datetime import datetime

# Add the parent app directory to the path so we can import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.prompt_manager import PromptManager, PromptVariant, PromptType

def create_comprehensive_prompts():
    """Create one comprehensive YAML file with ALL prompt variants"""
    
    print("🚀 Creating TaskMiner Comprehensive Prompts Configuration")
    print("=" * 60)
    
    # Initialize prompt manager (this loads all built-in variants)
    prompt_manager = PromptManager()
    
    # Show what prompts were loaded
    variants = prompt_manager.list_variants()
    total_variants = sum(len(v) for v in variants.values())
    
    print(f"\n📋 Loaded {total_variants} prompt variants:")
    for prompt_type, variant_names in variants.items():
        print(f"\n{prompt_type.upper().replace('_', ' ')}:")
        for variant_name in variant_names:
            variant_info = prompt_manager.get_variant_info(
                PromptType(prompt_type), variant_name
            )
            print(f"  • {variant_name} (v{variant_info.version})")
            print(f"    Description: {variant_info.description}")
            print(f"    Author: {variant_info.author}")
    
    # Create THE comprehensive prompts file - single source of truth
    prompts_file = "prompts.yaml"
    prompt_manager.save_to_config(prompts_file)
    
    print(f"\n✅ ALL prompts saved to: {prompts_file}")
    print("   📌 This is your SINGLE SOURCE OF TRUTH for all prompt variants!")
    print("   📌 Baseline, experimental, and custom variants - all in one place!")
    
    # Create backups directory if it doesn't exist and save backup
    if not os.path.exists("backups"):
        os.makedirs("backups")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"backups/prompts_backup_{timestamp}.yaml"
    prompt_manager.save_to_config(backup_file)
    
    print(f"📦 Backup saved to: {backup_file}")
    
    # Show file structure preview
    print(f"\n📄 Structure of {prompts_file}:")
    print("-" * 50)
    
    with open(prompts_file, 'r') as f:
        lines = f.readlines()
        in_section = False
        section_count = 0
        
        for i, line in enumerate(lines[:30]):  # Show first 30 lines
            stripped = line.strip()
            
            # Track major sections
            if stripped and not stripped.startswith(' ') and ':' in stripped and section_count < 3:
                if section_count > 0:
                    print(f"  ...")
                print(f"{i+1:2d}: {line.rstrip()}")
                section_count += 1
                in_section = True
            elif in_section and stripped.startswith(' ') and ':' in stripped and section_count <= 2:
                print(f"{i+1:2d}: {line.rstrip()}")
                if stripped.count(' ') == 2:  # End of variant
                    in_section = False
            elif section_count <= 2:
                print(f"{i+1:2d}: {line.rstrip()}")
        
        if len(lines) > 30:
            print(f"... (showing structure preview of {len(lines)} total lines)")
    
    print("-" * 50)
    
    return prompts_file

def validate_comprehensive_prompts(config_file):
    """Validate that all prompts can be loaded correctly from the single configuration"""
    
    print(f"\n🔍 Validating comprehensive prompts from {config_file}")
    
    try:
        # Load prompts from the comprehensive file
        test_manager = PromptManager(config_path=config_file)
        
        # Get all available variants
        all_variants = test_manager.list_variants()
        
        print(f"✅ Successfully loaded {sum(len(v) for v in all_variants.values())} variants")
        
        # Test each prompt type and variant
        for prompt_type_str, variant_names in all_variants.items():
            prompt_type = PromptType(prompt_type_str)
            print(f"\n📋 Testing {prompt_type_str.replace('_', ' ')} variants:")
            
            for variant_name in variant_names:
                try:
                    prompt_template = test_manager.get_prompt_template(prompt_type, variant_name)
                    
                    # Test formatting with sample data
                    sample_vars = {
                        "user_id": "test@example.com",
                        "user_name": "Test User",
                        "email_contents": "Can you please send me the report by Friday?",
                        "retrieved_context": "No additional context."
                    }
                    
                    formatted_prompt = prompt_template.format(**sample_vars)
                    print(f"  ✅ {variant_name} - loads and formats correctly")
                    
                except Exception as e:
                    print(f"  ❌ {variant_name} - error: {e}")
                    return False
        
        print(f"\n✅ All prompts validated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Create the comprehensive prompts configuration"""
    
    try:
        print("🎯 TaskMiner: Creating Single Comprehensive Prompts File")
        print("=" * 60)
        
        # Create the comprehensive prompts file
        prompts_file = create_comprehensive_prompts()
        
        # Validate the created file
        if validate_comprehensive_prompts(prompts_file):
            print("\n🎉 Comprehensive prompts configuration created successfully!")
        else:
            print("\n❌ Prompts configuration failed validation")
            return 1

        
        print("\n" + "=" * 60)
        print("✅ SUCCESS: Comprehensive prompts file created!")
        print("=" * 60)
        
        print("\n🎯 What was created:")
        print(f"  📄 prompts.yaml - Your single source of truth for all prompts")
        print(f"  📄 backups/prompts_backup_*.yaml - Timestamped backup")
        
        print(f"\n💡 Usage Example:")
        print(f"""
# Load from comprehensive file
prompt_manager = PromptManager(config_path="prompts.yaml")

# Use any variant
rag_pipeline(
    user_id="user@example.com",
    user_name="User Name",
    email_id="email_123",
    detection_prompt_variant="enhanced_with_examples",
    extraction_prompt_variant="concise_extraction"
)
        """)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error creating comprehensive prompts: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
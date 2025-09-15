# TaskMiner Prompts Directory

This directory contains all prompt-related files for the TaskMiner system, providing a centralized location for prompt management, evaluation, and documentation.

## 📁 Directory Structure

```
app/prompts/
├── README.md                    # This file - overview and usage
├── prompts.yaml                 # 🎯 MAIN: All prompt variants in one file
├── create_prompts.py           # [PENDING] Script to generate/update prompts.yaml
├── validate_prompts.py         # [PENDING] Validation and testing utilities 
├── templates/                  # [PENDING] Prompt templates and examples
│   ├── task_detection.yaml    # [PENDING] Template for detection prompts
│   ├── task_extraction.yaml   # [PENDING] Template for extraction prompts
│   └── examples/               # [PENDING] Example prompt variants
├── backups/                    # Automatic backups (git-ignored)
│   ├── prompts_20250720_143022.yaml
```

## 🎯 Main Files

### `prompts.yaml` - Single Source of Truth
The comprehensive configuration file containing ALL prompt variants:
- Production/baseline prompts
- Experimental variants
- Custom domain-specific prompts
- Version history and metadata

### `create_prompts.py` - Prompt Generator
Script to create and update the prompts.yaml file:
```bash
cd app/prompts
python create_prompts.py
```

### `validate_prompts.py` - Validation Utilities
Tools to test and validate prompt configurations:
```bash
python validate_prompts.py --config prompts.yaml
```

## 🚀 Quick Start

1. **Generate initial prompts file:**
   ```bash
   cd app/prompts
   python create_prompts.py
   ```

2. **Use in your code:**
   ```python
   from core.prompt_manager import PromptManager
   
   # Auto-loads from app/prompts/prompts.yaml
   prompt_manager = PromptManager(config_path="prompts/prompts.yaml")
   ```

3. **Add custom variants:**
   ```python
   # Custom prompts are saved to prompts/prompts.yaml
   prompt_manager.register_variant(my_custom_variant)
   prompt_manager.save_to_config("prompts/prompts.yaml")
   ```

## 📋 Benefits of This Structure

- ✅ **Clear separation**: All prompt files in one location
- ✅ **Scalable**: Easy to add new prompt types and variants
- ✅ **Organized**: Templates, backups, evaluations in logical subdirectories
- ✅ **Version control**: Track prompt evolution over time
- ✅ **Team collaboration**: Clear structure for team members
- ✅ **Documentation**: Centralized prompt engineering knowledge

## 🔄 Workflow

1. **Development**: Create/modify prompts in `templates/` or directly in `prompts.yaml`
2. **Testing**: Use `validate_prompts.py` to test prompt variants
3. **Evaluation**: Run A/B tests, save results to `evaluations/` [pending validation]
4. **Production**: Deploy `prompts.yaml` as single configuration file
5. **Backup**: Automatic backups created in `backups/` directory

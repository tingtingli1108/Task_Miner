"""
Simple script to display all available LLM models in a table format
"""

from llm_manager import display_models_dataframe

if __name__ == "__main__":
    # print("\nAvailable Models for TaskMiner")
    # print("=" * 120)
    # print(display_models_dataframe(available_only=True))
    print("\nAll models (including those without API keys):")
    print("-" * 60)
    print(display_models_dataframe(available_only=False))
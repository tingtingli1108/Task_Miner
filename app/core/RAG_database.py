import os
import json
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import ValidationError
from pinecone import Pinecone
from openai import OpenAI as OpenAIClient # To avoid naming conflicts

# Import local modules
from core.database import get_email_by_id, add_task, update_email_has_task_flag, update_new_email, get_emails_by_ids
from core.models import Task 
from core.llm_manager import create_llm, get_default_model

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Client Initialization ---
# Load API keys from .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
#pinecone_env = os.getenv('PINECONE_ENVIRONMENT') # not required for Pinecone SDK v3.0+


# Add lazy loading function for default LLM
_default_llm = None

def _get_default_llm():
    """Lazy initialization of default LLM - only creates when needed"""
    global _default_llm
    if _default_llm is None:
        try:
            default_model_id = get_default_model()
            _default_llm = create_llm(default_model_id, temperature=0)
            logging.info(f"Initialized default LLM with model: {default_model_id}")
        except Exception as e:
            logging.warning(f"Failed to initialize LLM with llm_manager, falling back to hardcoded model (gpt-4.1): {e}")
            _default_llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key=api_key)
    return _default_llm

# Initialize the OpenAI Client for embeddings
openai_client = OpenAIClient(api_key=api_key)


# Initialize Pinecone
try:
    if not all([pinecone_api_key, pinecone_index_name]):
        raise ValueError("Pinecone API key or index name not found in .env file.")
    
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(pinecone_index_name)

    logging.info(f"Pinecone initialized successfully for index '{pinecone_index_name}'.")
except Exception as e:
    logging.error(f"Failed to initialize Pinecone: {e}")
    pinecone_index = None


# Import prompt management system
from core.prompt_manager import get_task_detection_prompt, get_task_extraction_prompt, get_retrieval_context_filter_prompt

# --- Helper Functions ---

def _create_embedding(text, verbose=False):
    """Creates a vector embedding for a given text using OpenAI."""
    try:
        response = openai_client.embeddings.create(
            input=text, 
            model="text-embedding-3-small", 
            dimensions=1536 # set explicitly to avoid errors
        )
        logging.info(f"OpenAI embedding created successfully for text: {text}")
        if verbose: # for debugging
            logging.info(f"OpenAI embedding response: {response}")
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to create embedding with OpenAI: {e}")
        return None

def _filter_retrieved_context_with_llm(primary_email_content, context_emails, llm):
    """
    Uses an LLM to filter a list of retrieved emails to determine which ones are helpful
    for task extraction from the primary email.
    """
    if not context_emails:
        return [], []

    helpful_emails = []
    helpful_email_ids = []
    prompt_template = get_retrieval_context_filter_prompt()
    chain = prompt_template | llm

    logging.info("--- Starting LLM-based context filtering ---")
    for i, email in enumerate(context_emails):
        email_id = email.get('email_id', f'Unknown_ID_{i}')
        logging.info(f"Filtering document {i+1}/{len(context_emails)}: ID {email_id}")

        response = chain.invoke({
            "primary_email": primary_email_content,
            "retrieved_email": email['full_content']
        })

        response_cleaned = response.content.strip().lower()
        is_helpful = response_cleaned == "true"
        logging.info(f"  LLM relevance check for {email_id}: '{response.content.strip()}' -> {'KEPT' if is_helpful else 'DISCARDED'}")

        if is_helpful:
            helpful_emails.append(email['full_content'])
            helpful_email_ids.append(email['email_id'])
        elif response_cleaned == "false":
            pass # Email is not helpful, skip it
        else:
            # Check for invalid responses, log error
            logging.error(f"LLM returned unexpected response for email {email['email_id']}: '{response.content}'. Assuming email is not helpful.")
            pass # assume email is not helpful
            
    logging.info(f"--- LLM filtering complete. Kept {len(helpful_emails)} of {len(context_emails)} documents. ---")
    return helpful_emails, helpful_email_ids

def check_for_tasks(email_contents, user_id, user_name, email_id, model_id=None, prompt_variant="production_baseline", evaluation_mode=False):
    """Checks if the email contains any actionable tasks assigned to the user."""
    
    # Create LLM instance from model_id
    if model_id:
        current_llm = create_llm(model_id)
        logging.info(f"Using specified model: {model_id}")
    else:
        current_llm = _get_default_llm()  # Lazy load only when needed
        logging.info("Using default LLM instance")
    
    # Get the prompt template for the specified variant
    prompt_template = get_task_detection_prompt(prompt_variant)
    
    # --- ADDED FOR DEBUGGING ---
    logging.info(f"\n--- TASK CHECK PROMPT (variant: {prompt_variant}) ---")
    logging.info(f"Email ID: {email_id}")
    logging.info(f"User: {user_name} ({user_id})")
    logging.info(f"LLM Model: {getattr(current_llm, 'model_name', 'unknown')}")
    # --- END DEBUGGING ADDITION ---

    chain = prompt_template | current_llm
    response = chain.invoke({"email_contents": email_contents, "user_id": user_id, "user_name": user_name})
    
    # --- ADDED FOR DEBUGGING ---
    logging.info(f"--- RAW LLM RESPONSE ---\n'{response.content}'\n------------------------")
    # --- END DEBUGGING ADDITION -
    
    has_task_bool = response.content.strip().lower().startswith("true")

    # Skip database updates in evaluation mode
    if evaluation_mode:
        logging.info(f"EVALUATION MODE: {evaluation_mode}")
        logging.info(f"Skipping database updates for email {email_id}")
    
    if not evaluation_mode:
        # Update the email record in DynamoDB with the classification result
        update_email_has_task_flag(user_id, email_id, has_task_bool)
        logging.info(f"Updated email {email_id} with has_task flag: {has_task_bool}")

        # Update the email record in DynamoDB with "new" column as 0
        update_new_email(user_id=user_id, email_id=email_id)
        logging.info(f"Updated email {email_id} with new flag: 0")
    else:
        logging.info(f"EVALUATION MODE: Skipping database updates for email {email_id}")
    
    return has_task_bool
    

def extract_tasks(email_contents, user_id, user_name, retrieved_context, model_id=None, prompt_variant="production_baseline"):
    """Extracts all actionable tasks from the email for the given user, using context."""
    
    # Create LLM instance from model_id
    if model_id:
        current_llm = create_llm(model_id)
        logging.info(f"Using specified model: {model_id}")
    else:
        current_llm = _get_default_llm()  # Lazy load only when needed
        logging.info("Using default LLM instance")
    
    # Get the prompt template for the specified variant
    prompt_template = get_task_extraction_prompt(prompt_variant)
    
    logging.info(f"--- TASK EXTRACTION (variant: {prompt_variant}) ---")
    logging.info(f"LLM Model: {getattr(current_llm, 'model_name', 'unknown')}")
    
    chain = prompt_template | current_llm
    response = chain.invoke({
        "user_id": user_id, 
        "user_name": user_name,
        "email_contents": email_contents,
        "retrieved_context": retrieved_context
    })
    return response.content


# --- Vector Store Functions ---

def add_email_to_vectorstore(user_id, email_record, namespace=False):
    """Creates an embedding and stores it in Pinecone with metadata."""
    if not pinecone_index:
        logging.error("Pinecone index not available. Cannot add vector.")
        return
        
    text_to_embed = f"Subject: {email_record.get('subject', '')}\nFrom: {email_record.get('sender', '')}\n\n{email_record.get('email_body', '')}"
    vector = _create_embedding(text_to_embed)
    if not vector:
        return

    # Prepare metadata to store alongside the vector
    # FIX from : Cast the timestamp from DynamoDB's Decimal type to a standard int.
    timestamp_val = email_record.get('timestamp', 0)
    if timestamp_val is None:
        timestamp_val = 0
        
    metadata = {
        "sender": email_record.get('sender', ''),
        "to": email_record.get('recipients_to', ''),
        "cc": email_record.get('recipients_cc', ''),
        "bcc": email_record.get('recipients_bcc', ''),
        "subject": email_record.get('subject', ''),
        "timestamp": int(timestamp_val)
    }
    try:
        if namespace:
            namespace_name = namespace
        else:
            namespace_name = user_id

        pinecone_index.upsert(vectors=[(email_record['email_id'], vector, metadata)], namespace=namespace_name)
        logging.info(f"Successfully upserted vector for email {email_record['email_id']} into namespace {user_id}")
    except Exception as e:
        logging.error(f"Failed to upsert vector to Pinecone: {e}")

# --- Main Pipeline Function ---


def rag_pipeline(
        user_id, user_name, email_id, verbose=False, # required inputs
        model_id=None, detection_prompt_variant="production_baseline", extraction_prompt_variant="production_baseline",  # parameters for core RAG pipeline
        similarity_threshold=0.7, top_k=5, llm_filter_retrieval=True, months_back=None, # parameters for vectorstore retrieval
        evaluation_mode=False, email_record=False, namespace=False # parameters for evaluation
        ):

    """
    Runs the RAG pipeline to extract tasks from an email stored in the database
    and saves them to the tasks table.
    
    Args:
        user_id (str): User identifier
        user_name (str): User display name 
        email_id (str): Email identifier
        verbose (bool): Enable verbose logging
        model_id (str): Model identifier (e.g., "openai:gpt-4.1", "anthropic:claude-sonnet-4")
                       If None, uses default model
        detection_prompt_variant (str): Prompt variant for task detection
        extraction_prompt_variant (str): Prompt variant for task extraction
        similarity_threshold (float): Minimum similarity threshold for vectorstore retrieval
        top_k (int): Maximum number of documents to retrieve from vectorstore
        llm_filter_retrieval (bool): If True, use an LLM to filter retrieved documents for relevance
        months_back (int or None): Number of months back to include in retrieved context. If None, includes all historical emails (default: None)
        ## parameters for evaluation:
        evaluation_mode (bool): If True, skip all database modifications (for evaluation only)
        email_record (dict or bool): Exclude by default - will fetch from database using email_id. For evaluation, can specify pre-loaded email record to use instead of fetching from database.
        namespace (str or bool): Custom namespace for vectorstore operations. If False, uses user_id as namespace
    Returns:
        dict: Comprehensive pipeline results including inputs, outputs, and extracted tasks
    """
    # Initialize result structure with input parameters
    result = {
        "pipeline_inputs": {
            "user_id": user_id,
            "user_name": user_name,
            "email_id": email_id,
            "model_id": model_id,
            "llm_model_name": None,  # Will be filled when LLM is created
            "detection_prompt_variant": detection_prompt_variant,
            "extraction_prompt_variant": extraction_prompt_variant,
            "similarity_threshold": similarity_threshold,
            "llm_filter_retrieval": llm_filter_retrieval,
            "months_back": months_back
        },
        "pipeline_outputs": {
            "pred_has_task": False,
            "pred_retrieved_ids": [],
            "extracted_tasks": [],
            "tasks_created_count": 0,
            "success": False, # checks if full pipeline ran successfully
            "error_message": None
        }
    }
    
    if verbose:
        logging.info(f"Running RAG pipeline for user {user_name} ({user_id}) with email ID: {email_id}")
        if model_id:
            logging.info(f"Using model: {model_id}")
    
    # Skip database operations in evaluation mode
    try:
        if not email_record:
            email_record = get_email_by_id(user_id, email_id)
            if not email_record:
                raise ValueError(f"Email with ID {email_id} not found for user {user_id}")
            
        email_contents = f"""
from: {email_record.get('sender', '')}
to: {email_record.get('recipients_to', '')}
cc: {email_record.get('recipients_cc', '')}
subject: {email_record.get('subject', '')}
date: {email_record.get('date_str', '')}

body: 
{email_record.get('email_body', '')}
"""
    except Exception as e:
        logging.error(f"Error fetching email from database: {e}")
        result["pipeline_outputs"]["error_message"] = str(e)
        return result

    # Get the actual model being used and its name
    if model_id:
        current_llm = create_llm(model_id)
        actual_model_id = model_id
    else:
        current_llm = _get_default_llm()
        actual_model_id = get_default_model()
    
    # Extract model name from LLM instance
    llm_model_name = getattr(current_llm, 'model_name', getattr(current_llm, 'model', 'unknown'))
    result["pipeline_inputs"]["model_id"] = actual_model_id
    result["pipeline_inputs"]["llm_model_name"] = llm_model_name
    
    # Check for tasks
    has_task = check_for_tasks(email_contents, user_id, user_name, email_id, actual_model_id, detection_prompt_variant, evaluation_mode)
    result["pipeline_outputs"]["pred_has_task"] = has_task
    
    if not has_task:
        logging.info(f"LLM determined no tasks were present for user {user_name}.")
        result["pipeline_outputs"]["success"] = True  # Successfully determined no tasks
        return result
    
    logging.info(f"Task detected for {user_name}. Proceeding with retrieval and extraction.")

    retrieved_context = "No additional context retrieved."
    retrieved_ids = []
    if pinecone_index:
        text_to_embed = f"Subject: {email_record.get('subject', '')}\nFrom: {email_record.get('sender', '')}\n\n{email_record.get('email_body', '')}"
        query_vector = _create_embedding(text_to_embed)
        if query_vector:
            try:
                current_email_timestamp = email_record.get('timestamp', 0)
                if current_email_timestamp is None:
                    current_email_timestamp = 0
                
                if not namespace:   # namespace is the user_id by default. Option to override for evaluation.
                    namespace = user_id
                
                if evaluation_mode: # if evaluation mode, don't apply any filters to the context retrieval
                    query_results = pinecone_index.query(
                        vector=query_vector, 
                        top_k=top_k, 
                        namespace=namespace
                    )
                else:
                    # Build filter based on months_back parameter
                    if months_back is not None:
                        # Calculate timestamp cutoff for months_back filter
                        seconds_per_month = 30 * 24 * 60 * 60  # Approximate: 30 days per month
                        cutoff_timestamp = int(current_email_timestamp) - (months_back * seconds_per_month)
                        
                        # Apply both temporal filters: before current email AND within months_back window
                        timestamp_filter = {
                            "$lt": int(current_email_timestamp),
                            "$gte": cutoff_timestamp
                        }
                        logging.info(f"Applying {months_back} month time window filter for context retrieval.")
                    else:
                        # Apply only the "before current email" filter (original behavior)
                        timestamp_filter = {"$lt": int(current_email_timestamp)}
                        logging.info("No time window limit - retrieving all historical emails as context.")
                    
                    query_results = pinecone_index.query(
                        vector=query_vector, 
                        top_k=top_k, 
                        namespace=namespace,
                        filter={"timestamp": timestamp_filter}
                    )
                
                # Filter the results based on the score
                all_matches = query_results['matches']
                if months_back is not None:
                    logging.info(f"Found {len(all_matches)} relevant matches within {months_back} months in {namespace}.")
                else:
                    logging.info(f"Found {len(all_matches)} relevant matches in {namespace}.")
                filtered_matches = [match for match in all_matches if match['score'] >= similarity_threshold]
                logging.info(f"Found {len(filtered_matches)} relevant matches above the similarity threshold of {similarity_threshold}.")
                #all_retrieved_ids = [match['id'] for match in query_results['matches']]
                all_retrieved_ids = [match['id'] for match in filtered_matches]
                logging.info(f"Pinecone returned {len(all_retrieved_ids)} raw IDs: {all_retrieved_ids}")
                retrieved_ids = [rid for rid in all_retrieved_ids if rid != email_id]
                logging.info(f"Filtered down to {len(retrieved_ids)} IDs for context: {retrieved_ids}")
                
                # Store retrieved IDs in result
                result["pipeline_outputs"]["pred_retrieved_ids"] = retrieved_ids
            
                context_docs = []
                if retrieved_ids:
                    context_emails = get_emails_by_ids(user_id, retrieved_ids)
                    
                    if llm_filter_retrieval:
                        # Use LLM to filter context emails
                        context_docs, filtered_ids = _filter_retrieved_context_with_llm(email_contents, context_emails, current_llm)
                        result["pipeline_outputs"]["pred_retrieved_ids"] = filtered_ids # Update with filtered IDs
                    else:
                        # Use all retrieved emails for context
                        for context_email in context_emails:
                            context_docs.append(context_email['full_content'])
                
                if context_docs:
                    retrieved_context = "\n---\n".join(context_docs)
                    logging.info(f"Successfully retrieved {len(context_docs)} context documents from Pinecone.")
                else:
                    logging.info("No relevant context documents were retrieved from Pinecone.")
            except Exception as e:
                logging.error(f"Failed to query Pinecone: {e}")
                context_docs = []
                if retrieved_ids:
                    context_emails = get_emails_by_ids(user_id, retrieved_ids)
                    for context_email in context_emails:
                        context_docs.append(context_email['full_content'])
                if context_docs:
                    retrieved_context = "\n---\n".join(context_docs)
                    logging.info(f"Successfully retrieved {len(context_docs)} context documents from Pinecone.")
                else:
                    logging.info("No relevant context documents were retrieved from Pinecone.")
            except Exception as e:
                logging.error(f"Failed to query Pinecone: {e}")

    # CAPTURE THE FINAL CONTEXT
    final_context_for_llm = f"Primary Email:\n{email_contents}\n\nRetrieved Context:\n{retrieved_context}"

    llm_response_str = extract_tasks(email_contents, user_id, user_name, retrieved_context, actual_model_id, extraction_prompt_variant)
    if verbose:
        logging.info(f"Raw LLM Extraction Response: {llm_response_str}")

    tasks_created_count = 0
    extracted_tasks = []
    try:
        if not llm_response_str or not llm_response_str.strip():
            logging.warning("LLM returned an empty response. Assuming no tasks found.")
            result["pipeline_outputs"]["success"] = True
            return result
            
        # Clean the response to handle markdown code blocks
        cleaned_response = llm_response_str.strip()
        
        # Remove markdown code blocks if present
        if "```json" in cleaned_response:
            # Extract content between ```json and ```
            start_marker = "```json"
            end_marker = "```"
            start_idx = cleaned_response.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
                end_idx = cleaned_response.find(end_marker, start_idx)
                if end_idx != -1:
                    cleaned_response = cleaned_response[start_idx:end_idx].strip()
        elif "```" in cleaned_response:
            # Handle generic code blocks
            lines = cleaned_response.split('\n')
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip() == "```":
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (not in_code_block and line.strip().startswith(('[', '{'))):
                    json_lines.append(line)
            if json_lines:
                cleaned_response = '\n'.join(json_lines)
        
        tasks_list = json.loads(cleaned_response)
        if not isinstance(tasks_list, list):
            tasks_list = [tasks_list]
        if not tasks_list:
            logging.info("Extraction resulted in no tasks.")
            result["pipeline_outputs"]["success"] = True
            return result
        
        for task_data in tasks_list:
            try:
                task = Task(**task_data)
                task_dict = task.model_dump()
                extracted_tasks.append(task_dict)
                
                # Skip database operations in evaluation mode
                if not evaluation_mode:
                    # Add task to database
                    new_task_id = add_task(
                        user_id=user_id, 
                        email_id=email_id, 
                        task_data=task_dict,
                        retrieved_context_ids=retrieved_ids,
                        final_context=final_context_for_llm
                    )
                    if new_task_id:
                        tasks_created_count += 1
                else:
                    # In evaluation mode, count the task as "created" for metrics but don't save to DB
                    tasks_created_count += 1
                    logging.info(f"EVALUATION MODE: Skipping database save for task: {task_dict.get('task_title', 'Unknown')}")
                    
            except ValidationError as e:
                logging.error(f"LLM returned invalid task data, skipping. Error: {e}. Data: {task_data}")
            except Exception as e:
                logging.error(f"Failed to add task to database. Error: {e}. Data: {task_data}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response from LLM: {llm_response_str}")
        logging.error(f"Cleaned JSON: {cleaned_response}")
        result["pipeline_outputs"]["error_message"] = f"JSON parsing failed: {str(e)}"
        return result

    # Update final results
    result["pipeline_outputs"]["extracted_tasks"] = extracted_tasks
    result["pipeline_outputs"]["tasks_created_count"] = tasks_created_count
    result["pipeline_outputs"]["success"] = True
    
    logging.info(f"Pipeline complete. Successfully created {tasks_created_count} new task(s).")
    logging.info(f"Output of pipeline: {result}")
    return result


# --- Backward compatibility wrapper ---
## Discuss with team if this is needed
def rag_pipeline_simple(user_id, user_name, email_id, verbose=False, model_id=None, detection_prompt_variant="production_baseline", extraction_prompt_variant="production_baseline", similarity_threshold=0.7, months_back=None, evaluation_mode=False):
    """
    Backward compatibility wrapper that returns just the task count like the original function.
    
    Returns:
        int: Number of tasks created
    """
    result = rag_pipeline(user_id, user_name, email_id, verbose, model_id, detection_prompt_variant, extraction_prompt_variant, similarity_threshold, months_back, evaluation_mode)
    return result["pipeline_outputs"]["tasks_created_count"]
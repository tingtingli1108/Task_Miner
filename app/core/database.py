
import boto3
import uuid
from botocore.exceptions import ClientError
import logging
import os
from datetime import date, datetime, timedelta, timezone
import json 

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DynamoDB Client Initialization ---

# It's best practice to initialize the client once and reuse it.
# Boto3 will automatically use the credentials configured on the machine
# (either via 'aws configure' locally or the IAM Role on EC2).
try:
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    
    # Define table objects. Note: 'users' table is no longer needed for auth.
    emails_table = dynamodb.Table('task_miner_emails')
    tasks_table = dynamodb.Table('task_miner_tasks')
    
except Exception as e:
    logging.error(f"Could not connect to DynamoDB. Please check credentials and region. Error: {e}")
    dynamodb = None
    emails_table = None
    tasks_table = None

# --- S3 Uploads ---

def upload_to_s3(file, file_name, bucket, user_id):
    """Uploads a file to a user-specific folder in S3."""
    s3_client = boto3.client('s3')
    # All test uploads will now be inside the user  folder 
    s3_key = f"{user_id}/{os.path.basename(file_name)}" 
    
    try:
        s3_client.upload_fileobj(file, bucket, s3_key)
        logging.info(f"Successfully uploaded '{file_name}' to s3://{bucket}/{s3_key}")
        return True
    except ClientError as e:
        logging.error(f"Failed to upload to S3: {e}")
        return False
    except FileNotFoundError:
        logging.error(f"Local file not found: '{file_name}'. Please make sure it exists in the 'test' folder.")
        return False

# --- Email Management Functions ---

def add_email(user_id, parsed_email_data):
    """
    Adds a new, parsed email to the 'emails' table, using the Message-ID as the email_id.

    Args:
        user_id (str): The user's email/username from the authenticator.
        parsed_email_data (dict): A dictionary containing the parsed email fields.

    Returns:
        str: The email's Message-ID if successful, otherwise None.
    """
    if not emails_table:
        logging.error("Emails table not initialized.")
        return None
    
    # Use the unique Message-ID from the email header as our email_id
    email_id = parsed_email_data.get('message_id')
    if not email_id:
        logging.error("Cannot add email: Message-ID is missing from parsed data.")
        return None
    
    try:
        # Create the item to be stored. The 'email_id' sort key is now the 'message_id'.
        item_to_store = {
            'user_id': user_id,
            'email_id': email_id, # Using Message-ID as the sort key
            'sender': parsed_email_data.get('from'),
            'recipients_to': parsed_email_data.get('to'),
            'recipients_cc': parsed_email_data.get('cc'),
            'recipients_bcc': parsed_email_data.get('bcc'),
            'subject': parsed_email_data.get('subject'),
            'date_str': parsed_email_data.get('date'),
            'timestamp': parsed_email_data.get('timestamp'), # Unix Timestamp optoon
            'email_body': parsed_email_data.get('body'),
            'full_content': parsed_email_data.get('full_content'), # raw email,hard to read
            'new': 1
        }

        emails_table.put_item(Item=item_to_store)
        
        logging.info(f"Email with Message-ID {email_id} for user {user_id} added successfully.")
        return email_id
    except ClientError as e:
        logging.error(f"Error adding email for user {user_id}: {e.response['Error']['Message']}")
        return None

def get_email_by_id(user_id, email_id):
    """
    Fetches the full content of a single email by its ID (which is the Message-ID).

    Returns:
        dict: The email item if found, otherwise None.
    """
    if not emails_table:
        logging.error("Emails table not initialized.")
        return None
    try:
        response = emails_table.get_item(Key={'user_id': user_id, 'email_id': email_id})
        return response.get('Item')
    except ClientError as e:
        logging.error(f"Error fetching email {email_id} for user {user_id}: {e.response['Error']['Message']}")
        return None
    

def get_emails_by_ids(user_id, email_ids):
    """Fetches multiple email records from DynamoDB for a specific user."""
    if not emails_table or not email_ids:
        return []
    
    keys_to_get = [{'user_id': user_id, 'email_id': eid} for eid in email_ids]
    
    try:
        logging.info(f"Attempting to batch fetch from table: '{emails_table.name}'")
        
        # This is the corrected line. It calls batch_get_item on the main dynamodb resource.
        response = dynamodb.batch_get_item(
            RequestItems={
                emails_table.name: {
                    'Keys': keys_to_get
                }
            }
        )
        
        logging.info(f"Raw response from DynamoDB batch_get_item: {response}")
        
        return response.get('Responses', {}).get(emails_table.name, [])
    except ClientError as e:
        logging.error(f"Error batch getting emails for user {user_id}: {e.response['Error']['Message']}")
        return []
    


    
def get_new_emails(user_id):
    """
    Fetches the full content of all emails flagged as new to run through RAG pipeline

    Returns:
        List of JSON objects: [{'message-id': 'id1',...}, {'message-id': 'id2',...}]
    """
    if not emails_table:
        logging.error("Emails table not initialized.")
        return []

    try:
        query_params = {
            'KeyConditionExpression': boto3.dynamodb.conditions.Key('user_id').eq(user_id)
        }
        
        query_params['FilterExpression'] = boto3.dynamodb.conditions.Attr('new').eq(1)
             
        response = emails_table.query(**query_params)
        return response.get('Items', [])

    except ClientError as e:
        logging.error(f"Error querying tasks for user {user_id}: {e.response['Error']['Message']}")
        return []

def check_email_exists(user_id, message_id):
    """
    Checks if an email with a specific Message-ID already exists for a user
    using an efficient get_item call.

    Args:
        user_id (str): The user to check against.
        message_id (str): The Message-ID from the email header.

    Returns:
        bool: True if the email exists, False otherwise.
    """
    if not emails_table:
        logging.error("Emails table not initialized.")
        return True # Fail safely to prevent duplicates
    
    try:
        # Partition key is 'user_id' and sort key is 'email_id' (Message-ID)
        response = emails_table.get_item(
            Key={
                'user_id': user_id,
                'email_id': message_id
            }
        )
        
        # If 'Item' is in the response, it means the email exists.
        if 'Item' in response:
            logging.warning(f"Duplicate email found for user {user_id} with Message-ID: {message_id}")
            return True
        else:
            return False
            
    except ClientError as e:
        logging.error(f"Error checking for email existence for user {user_id}: {e.response['Error']['Message']}")
        return True # Fail safely
    
def update_email_has_task_flag(user_id, email_id, has_task):
    """
    Updates an email item in DynamoDB to set a 'has_task' flag.
    """
    if not emails_table:
        logging.error("Emails table not initialized.")
        return False
    try:
        has_task_number = 1 if has_task else 0
        emails_table.update_item(
            Key={'user_id': user_id, 'email_id': email_id},
            UpdateExpression="SET has_task = :val",
            ExpressionAttributeValues={':val': has_task_number}
        )
        logging.info(f"Updated email {email_id} with has_task = {has_task_number}.")
        return True
    except ClientError as e:
        logging.error(f"Error updating email {email_id} has_task flag: {e.response['Error']['Message']}")
        return False

def update_new_email(user_id, email_id):
    """
    Updates an email item in DynamoDB to set 'new' as 0, to signify it has been processed.
    """
    if not emails_table:
        logging.error("Emails table not initialized.")
        return False
    try:
        emails_table.update_item(
            Key={'user_id': user_id, 'email_id': email_id},
            UpdateExpression="SET #new_email = :val",
            ExpressionAttributeNames={'#new_email': 'new'},
            ExpressionAttributeValues={':val': 0}
        )
        logging.info(f"Updated email {email_id} with new = 0.")
        return True
    except ClientError as e:
        logging.error(f"Error updating email {email_id} with new flag: {e.response['Error']['Message']}")
        return False

# --- Task Management Functions ---
def add_task(user_id, email_id, task_data, retrieved_context_ids=[], final_context=""):
    """
    Adds a new task to the 'tasks' table.
    """
    if not tasks_table:
        logging.error("Tasks table not initialized.")
        return None
        
    task_id = str(uuid.uuid4())
    
    #due_date = task_data.get('due_date')
    # if isinstance(due_date, (date, datetime)):
    #     due_date = due_date.isoformat()

    try:
        item_to_store = {
                'user_id': user_id,
                'task_id': task_id,
                'email_id': email_id,
                'requester': task_data.get('requestor'),
                'task_title': task_data.get('task_title'),
                'task_description': task_data.get('description'),
                #'due_date': due_date, # fix test error. Ensure due_date is a string\
                'due_date': task_data.get('due_date'), # Ensure due_date is a strin
                'review_status': 'Pending',
                'task_status': 'To Do',
                'timestamp': int(datetime.now(timezone.utc).timestamp()),
                'retrieved_context_ids': ",".join(retrieved_context_ids), # Store as a comma-separated string
                'completion_status': task_data.get('completion_status', 0),
                'final_context': final_context # ADDED: Store the context used for generation
            
        }
        cleaned_item = {k: (v if v != "" else None) for k, v in item_to_store.items()}
        # ADDED LOGGING FOR DEBUGGING
        logging.info(f"--- SAVING TO TASKS TABLE ---\n{json.dumps(item_to_store, indent=2)}\n--------------------------")

        tasks_table.put_item(Item=cleaned_item)
        logging.info(f"Task {task_id} for user {user_id} added successfully.")
        return task_id
    except ClientError as e:
        logging.error(f"Error adding task for user {user_id}: {e.response['Error']['Message']}")
        return None


    
def get_tasks_by_user_and_status(user_id, review_status=None, task_status=None):
    """
    Fetches all tasks for a specific user, optionally filtering by review_status.
    """
    if not tasks_table:
        logging.error("Tasks table not initialized.")
        return []

    try:
        query_params = {
            'KeyConditionExpression': boto3.dynamodb.conditions.Key('user_id').eq(user_id)
        }
        
        if review_status:
            if task_status:
                query_params['FilterExpression'] = boto3.dynamodb.conditions.Attr('review_status').eq(review_status) & boto3.dynamodb.conditions.Attr('task_status').eq(task_status)
            else:    
                query_params['FilterExpression'] = boto3.dynamodb.conditions.Attr('review_status').eq(review_status)
        
        elif task_status:
            query_params['FilterExpression'] = boto3.dynamodb.conditions.Attr('task_status').eq(task_status)
            
        response = tasks_table.query(**query_params)
        return response.get('Items', [])

    except ClientError as e:
        logging.error(f"Error querying tasks for user {user_id}: {e.response['Error']['Message']}")
        return []

def get_tasks_within_last_k_days(user_id, k_days=30, only_active=False):
    """
    Gets all tasks for user within the last k_days and only active tasks if specified.

    Returns:
        List of JSON objects: [{'message-id': 'id1',...}, {'message-id': 'id2',...}]
    """
    if not tasks_table:
        logging.error("Tasks table not initialized.")
        return []

    try:
        query_params = {
            'KeyConditionExpression': boto3.dynamodb.conditions.Key('user_id').eq(user_id)
        }
        
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=k_days)
        timestamp_interval = [int(start_date.timestamp()), int(now.timestamp())]
        
        if k_days and only_active:
            query_params['FilterExpression'] = (boto3.dynamodb.conditions.Attr('timestamp').between(timestamp_interval[0], timestamp_interval[1]) & 
                                                boto3.dynamodb.conditions.Attr('review_status').eq('accepted') & 
                                                boto3.dynamodb.conditions.Attr('task_status').eq('To Do'))
        elif k_days:
            query_params['FilterExpression'] = boto3.dynamodb.conditions.Attr('timestamp').between(timestamp_interval[0], timestamp_interval[1])
        elif only_active:
            query_params['FilterExpression'] = (boto3.dynamodb.conditions.Attr('review_status').eq('accepted') & 
                                                boto3.dynamodb.conditions.Attr('task_status').eq('To Do'))
        
        response = tasks_table.query(**query_params)
        return response.get('Items', [])

    except ClientError as e:
        logging.error(f"Error querying tasks for user {user_id}: {e.response['Error']['Message']}")
        return []

def update_task_review_status(user_id, task_id, new_status):
    """
    Updates the review_status of a specific task.
    """
    if not tasks_table:
        logging.error("Tasks table not initialized.")
        return False
    try:
        tasks_table.update_item(
            Key={'user_id': user_id, 'task_id': task_id},
            UpdateExpression="SET review_status = :s",
            ExpressionAttributeValues={':s': new_status}
        )
        logging.info(f"Updated task {task_id} for user {user_id} to status {new_status}.")
        return True
    except ClientError as e:
        logging.error(f"Error updating task {task_id}: {e.response['Error']['Message']}")
        return False

def update_task_status(user_id, task_id, new_status):
    """
    Updates the task_status of a specific task.
    """
    if not tasks_table:
        logging.error("Tasks table not initialized.")
        return False
    try:
        tasks_table.update_item(
            Key={'user_id': user_id, 'task_id': task_id},
            UpdateExpression="SET task_status = :s",
            ExpressionAttributeValues={':s': new_status}
        )
        logging.info(f"Updated task {task_id} for user {user_id} to status {new_status}.")
        return True
    except ClientError as e:
        logging.error(f"Error updating task {task_id}: {e.response['Error']['Message']}")
        return False

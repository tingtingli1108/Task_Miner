import email
from email.policy import default
from email.header import decode_header
from email.utils import parsedate_to_datetime
import logging

# Import the database functions we'll need to interact with
from core.database import check_email_exists, add_email

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _decode_header_text(header_value):
    """Decodes email headers that might be in a non-ascii format."""
    if not header_value:
        return ""
    decoded_parts = decode_header(header_value)
    header_text = []
    for part, charset in decoded_parts:
        if isinstance(part, bytes):
            header_text.append(part.decode(charset or 'utf-8', errors='ignore'))
        else:
            header_text.append(str(part))
    return "".join(header_text)

def _get_email_body(msg_object):
    """Extracts the plain text body from an email.Message object."""
    if msg_object.is_multipart():
        for part in msg_object.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                try:
                    return part.get_payload(decode=True).decode('utf-8', errors='ignore')
                except:
                    return None
    else:
        try:
            return msg_object.get_payload(decode=True).decode('utf-8', errors='ignore')
        except:
            return None
    return ""

def parse_eml(file_content):
    """Parses the raw content of a .eml file into a structured dictionary."""
    try:
        msg = email.message_from_bytes(file_content, policy=default)
        timestamp = None
        date_str = msg.get('Date')
        if date_str:
            try:
                dt_object = parsedate_to_datetime(date_str)
                if dt_object:
                    timestamp = int(dt_object.timestamp())
            except Exception:
                logging.warning(f"Could not parse date string: {date_str}")

        parsed_data = {
            'from': _decode_header_text(msg.get('From')),
            'to': _decode_header_text(msg.get('To')),
            'cc': _decode_header_text(msg.get('Cc')),
            'bcc': _decode_header_text(msg.get('Bcc')),
            'subject': _decode_header_text(msg.get('Subject')),
            'date': _decode_header_text(date_str),
            'timestamp': timestamp,
            'message_id': msg.get('Message-ID'),
            'body': _get_email_body(msg),
            'full_content': file_content.decode('utf-8', errors='ignore'),
            'new': 1 # Add the new field with a default value of 1
        }

        if not parsed_data['body']:
            logging.error("Parsing failed: email body.")
            return None

        if not parsed_data['from']:
            logging.error("Parsing failed: Missing From header.")
            return None        
        if not parsed_data['to']:
            logging.error("Parsing failed: Missing To header.")
            return None
        
        if not parsed_data['subject']:
            logging.error("Parsing failed: Missing Subject header.")
            return None 
        
        if not parsed_data['date']:
            logging.error("Parsing failed: Missing Date header.")
            return None
        
        if not parsed_data['message_id'] :
            logging.error("Parsing failed: Missing Message-ID.")
            return None
    
            
        return parsed_data
    except Exception as e:
        logging.error(f"An error occurred during email parsing: {e}")
        return None

def process_and_store_email(user_id, file_content):
    """
    The main orchestrator function. It parses an email, checks for duplicates,
    and stores it in DynamoDB if it's new.
    """
    logging.info(f"Starting email processing for user: {user_id}")
    
    parsed_email = parse_eml(file_content)
    if not parsed_email:
        return ('ERROR', "Could not parse the provided .eml file.", None)
        
    message_id = parsed_email['message_id']

    logging.info(f"Checking for duplicate email with Message-ID: {message_id}")
    is_duplicate = check_email_exists(user_id=user_id, message_id=message_id)

    if is_duplicate:
        return ('DUPLICATE', "This email has already been processed.", None)

    logging.info(f"No duplicate found. Adding email to DynamoDB.")
    # The add_email function now returns the message_id, which we use as the email_id
    new_email_id = add_email(user_id=user_id, parsed_email_data=parsed_email)

    if new_email_id:
        return ('SUCCESS', f"Email successfully processed and stored with ID: {new_email_id}", new_email_id)
    else:
        return ('ERROR', "An error occurred while trying to save the email to the database.", None)
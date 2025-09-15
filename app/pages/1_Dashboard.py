import streamlit as st
import streamlit_authenticator as stauth
import time
from datetime import datetime
import yaml
from yaml.loader import SafeLoader
from core.database import get_tasks_by_user_and_status, update_task_review_status, update_task_status, upload_to_s3, get_new_emails
from core.email_parser import process_and_store_email
from core.RAG_database import add_email_to_vectorstore, rag_pipeline, get_email_by_id

color_scheme = {
    "background": "#1F2937",       # Dark Gray
    "surface": "#374151",          # Charcoal
    "primary": "#6366F1",          # Indigo
    "secondary": "#2DD4BF",        # Teal
    "accent": "#FDB515",           # Amber
    "text_primary": "#F9FAFB",     # Light Gray
    "text_muted": "#9CA3AF",       # Soft Gray
}

#Page Configuration
st.set_page_config(
    page_title="TaskMiner: Dashboard",
    page_icon="🤖",
    layout="wide"
)


#Sidebar Content
page_sidebar_header = f"""
<style>
[data-testid="stHeading"] {{
    color: {color_scheme['accent']};
    }}
</style>
"""
st.sidebar.header("Dashboards")
st.markdown(page_sidebar_header,unsafe_allow_html=True)

if st.sidebar.button("Task Decision Portal"):
    st.session_state.page = "TaskDecisionPortal"
if st.sidebar.button("Task Manager"):
    st.session_state.page = "TaskManager"

#Initialize Data Pipeline 
def full_data_pipeline(user_id, uploaded_files, n_files):
    
    placeholder = st.empty()

    with placeholder.container():
        st.markdown("""
                    <div style="
                        background-color: #374151;
                        margin-bottom: 40px;
                        padding: 10px 50px;
                        border-radius: 25px;
                        width: 100%;
                        overflow: hidden;
                        display: inline-block;
                        color: #F9FAFB;">
                    <h2 style='text-align:center;'>Processing, please wait...</h2>""", 
                    unsafe_allow_html=True)
        prog1 = st.progress(0, text="Step 1: Uploading to S3, Parsing, and Storing Email Content")
        prog2 = st.progress(0, text="Step 2: Storing New Emails In Vector Store")
        prog3 = st.progress(0, text="Step 3: Preparing RAG Pipeline")
        prog4 = st.progress(0, text="Step 4: Running RAG Pipeline on New Emails")
        
        for i, file in enumerate(uploaded_files):
            try:
                file_bytes = file.read()
            except ValueError as e:
                st.error(f"Could not read {file.name}: {e}")
        ## 1. Upload files to S3
            #Reset file pointer
            file.seek(0)
            upload_success = upload_to_s3(file=file,file_name=file.name, bucket= "task-miner-raw-emails", user_id=st.session_state.get("username"))
            #st.write(f"{file.name} was uploaded successfully!")
            
        ## 2. Parse raw .eml files and store in DynamoDB emails table
                        
            status = process_and_store_email(user_id=user_id, file_content=file_bytes)
            if isinstance(status, tuple):
                if status[0] == "DUPLICATE":
                    st.error(f"{file.name} is a duplicate!")
                    time.sleep(3)
                else:
                    percent_complete = int(((i + 1)/ n_files) * 100)
                    prog1.progress(percent_complete, text="Step 1: Uploading to S3, Parsing, and Storing Email Content")
        
        ## 3. Retrieve all new emails from DynamoDB emails table
        new_emails = get_new_emails(user_id=user_id)
        if new_emails:
            sorted_emails = sorted(new_emails, key=lambda x: datetime.strptime(x['date_str'], "%a, %d %b %Y %H:%M:%S %z"), reverse=True)
            #sorted_emails = sorted(new_emails, key=lambda x: safe_parse_due_date(x['date_str']), reverse=True)
            
        ## 4. Store in vectorstore
            for i, email in enumerate(sorted_emails):
                add_email_to_vectorstore(user_id=user_id,email_record=email)
                percent_complete = int(((i + 1)/ len(sorted_emails)) * 100)
                prog2.progress(percent_complete, text="Step 2: Storing New Emails In Vector Store")

        ## 4.5 Allow at least 30 seconds for Vector Store to update
            for i in range(100):
                time.sleep(0.03)
                prog3.progress(i+1, text="Step 3: Preparing RAG Pipeline")

        ## 5. Run RAG pipeline function
            for i, email in enumerate(sorted_emails):
                percent_complete = int(((i + 1)/ len(sorted_emails)) * 100)
                rag_pipeline(user_id= user_id, user_name=st.session_state.get("name"), email_id=email["email_id"], verbose=True, model_id='openai:gpt-4o-mini', top_k=5, similarity_threshold=.7, extraction_prompt_variant='few_shots_extraction')
                prog4.progress(percent_complete, text="Step 4: Running RAG Pipeline on New Emails")
    time.sleep(1)
    placeholder.empty()

#Requirements to clear the file uploader
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def update_file_attachment_key():
    st.session_state.uploader_key += 1

#Details Popup
@st.dialog("Related Emails")
def details_popup(user_id,task):
    #Get original email content
    original_email = get_email_by_id(user_id, task['email_id'])
    if original_email:
        related_contents = [original_email]
    else:
        related_contents = []

    if task['retrieved_context_ids']:
        retrieved_context_ids = task['retrieved_context_ids'].split(',')
        if isinstance(retrieved_context_ids, list):
            for related_id in retrieved_context_ids:
                current_email = get_email_by_id(user_id, related_id)
                if current_email:
                    related_contents.append(current_email)

    if related_contents:
        subject_to_email = {f"{i+1}: {email['subject']}": email for i, email in enumerate(related_contents)}
        selected_subject = st.selectbox("Select an email", options=list(subject_to_email.keys()))
        selected_email = subject_to_email[selected_subject]
        st.write(f"{selected_email['email_body']}")
    else:
        st.markdown("<h2 style='text-align:center;'>There were no related emails found.</h2>", unsafe_allow_html=True)

#Load due_dates safely
def safe_parse_due_date(due_date_str):
    try:
        return datetime.strptime(due_date_str, "%Y-%m-%d")
    except Exception:
        return datetime.min

#Load authenticator
#Initialize Authenticator
if 'authenticator' not in st.session_state:
    with open('users/config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    st.session_state.authenticator = authenticator
else:
    authenticator = st.session_state.authenticator


#Login Widget
try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state.get('authentication_status'):
    authenticator.logout(use_container_width=True)
    #Initalize Page
    if "page" not in st.session_state:
        st.session_state.page = "TaskDecisionPortal"
    #Task Decision Portal

    #Initialize tasks pending approval in session_state once (this would be pulled from DynamoDB)
    if "tasks_pending_approval" not in st.session_state:
        st.session_state['tasks_pending_approval'] = get_tasks_by_user_and_status(st.session_state.get("username"), review_status='Pending')
    tasks_pending_approval = st.session_state.tasks_pending_approval
    
    #Initialize tasks being tracked (this would be pulled from DynamoDB)
    if "tracked_tasks" not in st.session_state:
        st.session_state['tracked_tasks'] = get_tasks_by_user_and_status(st.session_state.get("username"), review_status='accepted', task_status='To Do')
    tracked_tasks = st.session_state.tracked_tasks
    
    #st.write(tasks_pending_approval)
    #st.write(tracked_tasks)

    if st.session_state.page == "TaskDecisionPortal":
        st.markdown("""<h1 style="text-align: center;">Task Decision Portal</h1>""", unsafe_allow_html=True)
        st.write(f'### Welcome *{st.session_state.get("name")}*,')

        sort_dropdown = st.selectbox(label="Identified_Email_Sorting", options=("Due Date Ascending", "Due Date Descending", "Identified Date Ascending", "Indentified Date Descending"), accept_new_options=False, label_visibility="hidden")

        if not tasks_pending_approval:
            st.markdown(
                    """
                    <div style="
                        background-color: #374151;
                        margin-bottom: 40px;
                        padding: 10px 50px;
                        border-radius: 25px;
                        width: 100%;
                        overflow: hidden;
                        display: inline-block;
                        color: #F9FAFB;">
                        <h4 style="text-align: center;">There are no identified tasks to track.<br> 
                        To identify more tasks please upload more emails through the <b><i>Email Upload</i></b> below.</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        #Sort based on selection
        if sort_dropdown == "Due Date Ascending":
            sorted_tasks_pending_approval = sorted(
                                                tasks_pending_approval,
                                                key=lambda x: safe_parse_due_date(x.get('due_date')) if x.get('due_date') else datetime.min,
                                                reverse=False
                                            )
        elif sort_dropdown == "Due Date Descending":
            sorted_tasks_pending_approval = sorted(
                                                tasks_pending_approval,
                                                key=lambda x: safe_parse_due_date(x.get('due_date')) if x.get('due_date') else datetime.min,
                                                reverse=True
                                            )
        elif sort_dropdown == "Identified Date Ascending":
            sorted_tasks_pending_approval = sorted(
                                                tasks_pending_approval,
                                                key=lambda x: datetime.fromtimestamp(float(x['timestamp'])) if x.get('timestamp') else datetime.min,
                                                reverse=False
                                            )
        else:
            sorted_tasks_pending_approval = sorted(
                                                tasks_pending_approval,
                                                key=lambda x: datetime.fromtimestamp(float(x['timestamp'])) if x.get('timestamp') else datetime.min,
                                                reverse=True
                                            )

        for i, task in enumerate(sorted_tasks_pending_approval):
            with st.container():
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(
                    f"""
                    <div style="
                        background-color: #374151;
                        margin-bottom: 10px;
                        padding: 10px 50px;
                        padding-top: 10px;
                        padding-bottom: 20px;
                        border-radius: 25px;
                        width: 100%;
                        height: 100%;
                        overflow: hidden;
                        display: inline-block;
                        color: #F9FAFB;">
                        <h3>{task["task_title"]}</h3>
                        {task["task_description"]}<br>
                        Requestor: {task.get("requester", 'Unknown')}<br>
                        Due Date: {task.get('due_date', 'N/A')}<br>
                        Identified Date: {datetime.fromtimestamp(float(task.get('timestamp'))) if task.get('timestamp') else datetime.min}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                    
                with col2:
                    st.markdown("<div style='height: 50px;'>", unsafe_allow_html=True)
                    if st.button("✅ Accept", key=f"accept_{i}"):
                        update_success = update_task_review_status(user_id = st.session_state.get("username"), task_id = task["task_id"], new_status='accepted')
                        if update_success:
                            message = st.success(f"Accepted: {task['task_title']}")
                            time.sleep(1)
                            message.empty()
                            #Update tasks
                            st.session_state['tasks_pending_approval'] = get_tasks_by_user_and_status(st.session_state.get("username"), review_status='Pending')
                            st.session_state['tracked_tasks'] = get_tasks_by_user_and_status(st.session_state.get("username"), review_status='accepted', task_status='To Do')
                            st.rerun()
                        else:
                            message = st.success(f"There was an error accepting: {task['task_title']}")
                            time.sleep(1)
                            message.empty()

                    if st.button("❌ Decline", key=f"decline_{i}"):
                        update_success = update_task_review_status(user_id = st.session_state.get("username"), task_id = task["task_id"], new_status='declined')
                        if update_success:
                            message = st.error(f"Declined: {task['task_title']}")
                            time.sleep(1)
                            message.empty()
                            #Update tasks
                            st.session_state['tasks_pending_approval'] = get_tasks_by_user_and_status(st.session_state.get("username"), review_status='Pending')
                            st.session_state['tracked_tasks'] = get_tasks_by_user_and_status(st.session_state.get("username"), review_status='accepted', task_status='To Do')
                            st.rerun()
                        else:
                            message = st.success(f"There was an error declining: {task['task_title']}")
                            time.sleep(1)
                            message.empty()
        
        
        st.markdown("""<h1 style="text-align: center;">Email Upload</h1>""", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload emails in EML format", type=["eml"], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
        
        # --- Pipeline trigger ---
        if uploaded_files:
            if st.button("✅ Upload", key="Upload"):
                full_data_pipeline(user_id=st.session_state.get("username"), uploaded_files=uploaded_files, n_files=len(uploaded_files))

        ## 6. Rerun session to display new tasks identified
                st.session_state['tasks_pending_approval'] = get_tasks_by_user_and_status(st.session_state.get("username"), review_status='Pending')
                st.session_state['tracked_tasks'] = get_tasks_by_user_and_status(st.session_state.get("username"), review_status='accepted', task_status='To Do')
                
                #Reset file_uploader and rerun page
                update_file_attachment_key()
                st.rerun()
    
    elif st.session_state.page == "TaskManager":
        st.markdown("""<h1 style="text-align: center;">Task Manager</h1>""", unsafe_allow_html=True)
        st.write(f'### Welcome *{st.session_state.get("name")}*,')

        if not tracked_tasks:
            st.markdown(
                    """
                    <div style="
                        background-color: #374151;
                        margin-bottom: 40px;
                        padding: 10px 50px;
                        border-radius: 25px;
                        width: 100%;
                        overflow: hidden;
                        display: inline-block;
                        color: #F9FAFB;">
                        <h4 style="text-align: center;">There are no tasks to track. Well Done!<br> 
                        To identify more tasks please upload more emails through the <b><i>Email Upload</i></b> page.</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        for i, task in enumerate(tracked_tasks):
            with st.container():
                col3, col4 = st.columns([6, 1])
                with col3:
                    st.markdown(
                    f"""
                    <div style="
                        background-color: #374151;
                        margin-bottom: 40px;
                        padding: 10px 50px;
                        padding-top: 10px;
                        padding-bottom: 20px;
                        border-radius: 25px;
                        width: 100%;
                        height: 100%;
                        overflow: hidden;
                        display: inline-block;
                        color: #F9FAFB;">
                        <h3>{task["task_title"]}</h3>
                        {task["task_description"]}<br>
                        Requestor: {task.get("requester", 'Unknown')}<br>
                        Due Date: {task.get('due_date', 'N/A')}<br>
                        Identified Date: {datetime.fromtimestamp(float(task.get('timestamp'))) if task.get('timestamp') else datetime.min}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                with col4:
                    st.markdown("<div style='height: 50px;'>", unsafe_allow_html=True)
                    if st.button("✅ Mark as Complete", key=f"complete_{i}"):
                        update_success = update_task_status(user_id = st.session_state.get("username"), task_id = task["task_id"], new_status='completed')
                        if update_success:
                            message = st.success(f"Completed: {task['task_title']}")
                            time.sleep(2)
                            message.empty()
                            #Update tasks
                            st.session_state['tasks_pending_approval'] = get_tasks_by_user_and_status(st.session_state.get("username"), review_status='Pending')
                            st.session_state['tracked_tasks'] = get_tasks_by_user_and_status(st.session_state.get("username"), review_status='accepted', task_status='To Do')
                            st.rerun()
                        else:
                            message = st.success(f"There was an error declining: {task['task_title']}")
                            time.sleep(1)
                            message.empty()
                    if st.button("🔍 View Related Emails", key=f"related_emails_{task['task_id']}"):
                        details_popup(user_id = st.session_state.get("username"),task=task)
        
elif st.session_state.get('authentication_status') is False:
    st.error('Username/password is incorrect')
elif st.session_state.get('authentication_status') is None:
    st.warning('Please enter your username and password')

import os
import io
import zipfile

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Variables to downloar from google drive
GOOGLE_DRIVE_FOLDER_ID = "1EYgJrhf3BKHypPQLT5xwTHhsHa2BYMFt"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly'] # Read-only access is sufficient for downloading

def authenticate_google_drive():
    """Authenticates with Google Drive API and returns the service object."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('drive', 'v3', credentials=creds)
    return service

def confirm_if_file_is_on_drive(zip_file_name):
    """
    Confirms the file name is correct by finding it in the google drive.
    
    Args:
        zip_file (str): the supposed name of the zip file to be found.
    
    Returns:
        Optional[int]: The file id, if file is found, otherwise None.
    """
    service = authenticate_google_drive()
    folder_id = GOOGLE_DRIVE_FOLDER_ID

    file_id = None
    # Search for the file within the specified folder
    #query = f"'{folder_id}' in parents and name = '{zip_file_name}' and mimeType = 'application/zip'"
    query = f"'{folder_id}' in parents and trashed = false"

    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)',
        supportsAllDrives=True, 
        includeItemsFromAllDrives=True 
    ).execute()
    items = results.get('files', [])
    index = None
    
    for i, item in enumerate(items):
        if item['name'] == zip_file_name:
            index = i
            break
 
    if index:
        return {
            "file_id": items[index]["id"],
            "file_name": items[index]["name"]
            }
    else:
        return None

def download_and_extract_zip_from_drive(file_id, download_path=".", extract_path="."):
    """
    Downloads a specified file id from a Google Drive folder and extracts it.

    Args:
        file_id (str): The id of the file to download.
        download_path (str): The local path where the ZIP file will be downloaded.
        extract_path (str): The local path where the contents of the ZIP file will be extracted.
    """
    service = authenticate_google_drive()
    
    # Create download and extract directories if they don't exist
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

        # Download the file
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()

    # Save the downloaded ZIP file
    local_zip_filepath = os.path.join(download_path, "CSV_FILES.zip")
    with open(local_zip_filepath, 'wb') as f:
        fh.seek(0)
        f.write(fh.read())

    # Extract the ZIP file

    try:
        with zipfile.ZipFile(local_zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        contents = os.listdir(extract_path)
        return { "sucess": contents}
    except zipfile.BadZipFile:
        return { "error": f"The downloaded file is not a valid ZIP file."}
    except Exception as e:
        return { "error": f"An error occurred during extraction: {e}"}


def extract(local_zip_filepath, extract_path):
    try:
        with zipfile.ZipFile(local_zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        contents = os.listdir(extract_path)
        return { "sucess": contents}
    except zipfile.BadZipFile:
        return { "error": f"The downloaded file is not a valid ZIP file."}
    except Exception as e:
        return { "error": f"An error occurred during extraction: {e}"}

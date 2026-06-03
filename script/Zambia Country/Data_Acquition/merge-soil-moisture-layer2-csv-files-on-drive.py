import pandas as pd
import io
import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from icecream import ic
ic.configureOutput(prefix=f'Debug | ', includeContext=True)
# -------------------------------
# CONFIGURATION
# -------------------------------
DRIVE_FOLDER_ID = '1lRuvS1EYg8mkjoP9AeTe3hHuXlwYoiFW'  # e.g., '1abc123...' from the folder URL
LOCAL_TEMP_DIR = './temp_csvs'            # temporary folder to download files
MASTER_CSV_NAME = 'master_southern_province_soil-moisture-layer2.csv'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 
          'https://www.googleapis.com/auth/drive.file']

# -------------------------------
# AUTHENTICATE AND BUILD DRIVE SERVICE
# -------------------------------
def authenticate_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

# -------------------------------
# DOWNLOAD ALL CSV FILES FROM A FOLDER
# -------------------------------
def download_csvs_from_drive(service, folder_id, local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    # List all files in the folder
    query = f"'{folder_id}' in parents and mimeType='text/csv'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    
    ic(files)
    downloaded_files = []
    for file in files:
        print(f"Downloading {file['name']}...")
        request = service.files().get_media(fileId=file['id'])
        fh = io.FileIO(os.path.join(local_dir, file['name']), 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.close()
        downloaded_files.append(os.path.join(local_dir, file['name']))
    return downloaded_files

## -------------------------------
## MERGE ALL CSV FILES INTO ONE DATAFRAME
## -------------------------------
def merge_csv_files(file_paths):
    # We'll collect dataframes in a list
    all_dfs = []
    
    target_columns = ['soil_moisture_layer2', 'date', 'day', 'district', 'month', 'year']
    
    for path in file_paths:
        df = pd.read_csv(path)
        print(f"Loaded {path}: {len(df)} rows")
        
        # Standardise column names (lowercase, strip spaces)
        df.columns = df.columns.str.lower().str.strip()
        df = df.dropna(how='all')
        
        if 'district' in df.columns and 'soil_moisture_layer2' in df.columns:
            df = df.dropna(subset=['district', 'soil_moisture_layer2'])

        existing_cols = [c for c in target_columns if c in df.columns]
        df = df[existing_cols]
        all_dfs.append(df)

    master_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        
    return master_df

## -------------------------------
## UPLOAD MASTER CSV TO DRIVE
## -------------------------------
def upload_to_drive(service, local_file_path, folder_id):
    file_metadata = {
        'name': os.path.basename(local_file_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(local_file_path, mimetype='text/csv')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded to Drive with file ID: {file.get('id')}")

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == '__main__':
    # 1. Authenticate
    drive_service = authenticate_drive()
    
    # 2. Download all CSVs from the Drive folder
    csv_files = download_csvs_from_drive(drive_service, DRIVE_FOLDER_ID, LOCAL_TEMP_DIR)
    
    if not csv_files:
        print("No CSV files found in the folder.")
    else:
        # 3. Merge all into one DataFrame
        master_df = merge_csv_files(csv_files)
        print(f"\nMerged DataFrame has {len(master_df)} rows and {len(master_df.columns)} columns.")
        ic(master_df)
        
        # 4. Save master CSV locally
        local_master_path = os.path.join(LOCAL_TEMP_DIR, MASTER_CSV_NAME)
        master_df.to_csv(local_master_path, index=False)
        print(f"Saved master CSV locally: {local_master_path}")
        
        # 5. Upload the master CSV back to Drive (optional)
        upload_to_drive(drive_service, local_master_path, DRIVE_FOLDER_ID)
        
        # 6. Clean up temporary files (optional)
        # import shutil
        # shutil.rmtree(LOCAL_TEMP_DIR)

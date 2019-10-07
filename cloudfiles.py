import pathlib
from google.cloud import storage

def dl_from_gcs(creds, bucket, filepath, localfolder=''):
    """
    Download file from gcs to local.

    Args:
        creds (string): Location of the GCS credentials file.  
        bucket (string): Google cloud bucket name
        filepath (string): Location of the file within the google bucket
        localfolder (string): Local folder to download to
        """


    #https://cloud.google.com/docs/authentication/production#auth-cloud-implicit-python
    creds = pathlib.Path(creds)
    filepath = pathlib.Path(filepath)

    if creds.is_file() is False:
        raise Exception('Credentials not found.')

    client = storage.Client.from_service_account_json(creds)
    bucket = client.get_bucket(bucket)
    blob = bucket.get_blob(str(filepath))

    if blob is None:
        raise Exception('File not found.')
    else:
        dl_path = pathlib.Path(localfolder) / pathlib.Path(filepath.name)
        print(dl_path)
        blob.download_to_filename(dl_path)

    # if want the result in memory, can use:
    # blob.download_as_string

def files_in_folder(creds, bucket, folder):
    """
    Return a list of files in a folder on gcs.

    Args:
        creds (string): Location of the GCS credentials file. 
            https://cloud.google.com/docs/authentication/production#auth-cloud-implicit-python
        bucket (string): Google cloud bucket name
        folder (string): Location of the folder within the google bucket
    """
    creds = pathlib.Path(creds)
    folder = pathlib.Path(folder)

    if creds.is_file() is False:
        raise Exception('Credentials not found.')

    client = storage.Client.from_service_account_json(creds)
    bucket = client.get_bucket(bucket)

    files = [blob.name for blob in bucket.list_blobs(prefix=folder)]

    return files

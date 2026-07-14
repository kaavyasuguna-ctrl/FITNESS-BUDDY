#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import ibm_boto3
import time
from tests.utils import credential_store
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from ibm_botocore.config import Config

config = Config(max_pool_connections=100)

cos_credentials = credential_store.cos_credentials

client_cos = ibm_boto3.client(
    service_name="s3",
    endpoint_url=cos_credentials["endpoint_url"],
    aws_access_key_id=cos_credentials["cos_hmac_keys"]["access_key_id"],
    aws_secret_access_key=cos_credentials["cos_hmac_keys"]["secret_access_key"],
    config=config,
)

buckets = client_cos.list_buckets()
searched_strings_in_name = [
    # ONLY SDK BUCKETS
    "qa-sdk-tests-",
    "basesdktestproject",
    "text-extraction-sdk-tests-",
    "watsonx-ai-sdk-test-pypi-",
    "wx-autoai-sdk-tests-",
    "wx-autoai-tests-",
    # OTHERS BUCKETS WE DO NOT INCLUDE
    # "autoai-tests-",
    # "autoairagsmoketestproject-donotdelete-",
    # "autoairagtestproject-",
    # "autoairagtestingproject-",
    # "autocreatedprojectqa",
    # "autocreatedragproject",
    # "car-rental-",
    # "churn-",
    # "credit-risk-",
    # "product-line-",
    # "wml-autoai-tests-",

]

named_bucket_list = []
bucket_max_age = timedelta(days=3)
cutoff_datetime = datetime.now(timezone.utc) - bucket_max_age

for bucket in buckets["Buckets"]:
    bucket_name = bucket["Name"]
    creation_date = bucket["CreationDate"]
    print("Bucket Name: {0}".format(bucket_name))
    if any(bucket_name.startswith(s) for s in searched_strings_in_name):
        if creation_date <= cutoff_datetime:
            named_bucket_list.append(bucket_name)
        else:
            print(
                "SKIPPED {0}: created at {1}, newer than 3 days.".format(
                    bucket_name, creation_date
                )
            )

print("Found {0} buckets older than 3 days to delete.".format(len(named_bucket_list)))

MAX_BUCKET_WORKERS = 5
MAX_OBJECT_WORKERS = 10


def get_error_response(exception: Exception) -> dict[str, Any]:
    return cast(dict[str, Any], getattr(exception, "response", {}))


def delete_object(bucket_name, file):
    try:
        client_cos.delete_object(Bucket=bucket_name, Key=file["Key"])
        print("Deleted item: {0} ({1} bytes).".format(file["Key"], file["Size"]))
    except Exception as e:
        error_response = get_error_response(e)
        error_code = error_response.get("Error", {}).get("Code")
        if error_code == "NoSuchKey":
            print("SKIPPED item {0}: already deleted.".format(file["Key"]))
        else:
            raise

def wait_until_deleted(bucket_name, retries=5, delay=1):
    for _ in range(retries):
        if not bucket_exists(bucket_name):
            return True
        time.sleep(delay)
    return False

def abort_multipart_upload(bucket_name, upload):
    try:
        client_cos.abort_multipart_upload(
            Bucket=bucket_name, Key=upload["Key"], UploadId=upload["UploadId"]
        )
        print(
            "Aborted multipart upload: {0} (UploadId: {1})".format(
                upload["Key"], upload["UploadId"]
            )
        )
    except Exception as e:
        error_response = get_error_response(e)
        error_code = error_response.get("Error", {}).get("Code")
        if error_code == "NoSuchUpload":
            print(
                "SKIPPED multipart upload {0}: already aborted.".format(upload["Key"])
            )
        else:
            raise


def get_all_objects(bucket_name):
    """Paginate through all objects in a bucket to avoid the 1000 object limit."""
    all_files = []
    marker = None

    while True:
        kwargs = {"Bucket": bucket_name}
        if marker:
            kwargs["Marker"] = marker

        response = client_cos.list_objects(**kwargs)
        all_files.extend(response.get("Contents", []))

        if response.get("IsTruncated"):
            marker = response["NextMarker"]
        else:
            break

    return all_files


def get_all_multipart_uploads(bucket_name):
    """Paginate through all multipart uploads."""
    all_uploads = []
    key_marker = None
    upload_id_marker = None

    while True:
        kwargs = {"Bucket": bucket_name}
        if key_marker:
            kwargs["KeyMarker"] = key_marker
            kwargs["UploadIdMarker"] = upload_id_marker

        response = client_cos.list_multipart_uploads(**kwargs)
        all_uploads.extend(response.get("Uploads", []))

        if response.get("IsTruncated"):
            key_marker = response["NextKeyMarker"]
            upload_id_marker = response["NextUploadIdMarker"]
        else:
            break

    return all_uploads


def bucket_exists(bucket_name):
    """Check if a bucket actually exists."""
    try:
        client_cos.head_bucket(Bucket=bucket_name)
        return True
    except Exception as e:
        error_response = get_error_response(e)
        error_code = error_response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchBucket"):
            return False
        raise


def delete_bucket(bucket_name):
    try:
        # Pre-check: skip entirely if already gone
        if not bucket_exists(bucket_name):
            return bucket_name, None, "skipped"

        # Step 1: collect ALL objects first (paginated)
        all_files = get_all_objects(bucket_name)
        print(
            "Bucket {0}: found {1} objects to delete.".format(
                bucket_name, len(all_files)
            )
        )

        # Step 2: delete all objects in parallel and WAIT for full completion
        if all_files:
            with ThreadPoolExecutor(max_workers=MAX_OBJECT_WORKERS) as obj_executor:
                obj_futures = [
                    obj_executor.submit(delete_object, bucket_name, file)
                    for file in all_files
                ]
                for future in as_completed(obj_futures):
                    future.result()

        # Step 3: collect ALL multipart uploads (paginated)
        all_uploads = get_all_multipart_uploads(bucket_name)
        print(
            "Bucket {0}: found {1} multipart uploads to abort.".format(
                bucket_name, len(all_uploads)
            )
        )

        # Step 4: abort all multipart uploads in parallel and WAIT for full completion
        if all_uploads:
            with ThreadPoolExecutor(max_workers=MAX_OBJECT_WORKERS) as mp_executor:
                mp_futures = [
                    mp_executor.submit(abort_multipart_upload, bucket_name, upload)
                    for upload in all_uploads
                ]
                for future in as_completed(mp_futures):
                    future.result()

        # Step 5: only delete bucket AFTER all objects and uploads are fully gone
        client_cos.delete_bucket(Bucket=bucket_name)

        # Step 6: verify the bucket is truly gone
        if not wait_until_deleted(bucket_name):
            return (
                bucket_name,
                "Bucket still exists after deletion — may have been recreated.",
                "failed",
            )

        return bucket_name, None, "deleted"

    except Exception as e:
        error_response = get_error_response(e)
        error_code = error_response.get("Error", {}).get("Code")
        error_message = error_response.get("Error", {}).get("Message", str(e))
        if error_code == "NoSuchBucket":
            print("SKIPPED {0}: already deleted.".format(bucket_name))
            return bucket_name, None, "skipped"
        return (
            bucket_name,
            "[{0}] {1}".format(error_code, error_message),
            "failed",
        )


# Summary counters
results = {"deleted": [], "skipped": [], "failed": []}

with ThreadPoolExecutor(max_workers=MAX_BUCKET_WORKERS) as executor:
    futures = {executor.submit(delete_bucket, el): el for el in named_bucket_list}

    for future in as_completed(futures):
        bucket_name, error, status = future.result()
        results[status].append(bucket_name)
        if status == "deleted":
            print("DELETED {0}".format(bucket_name))
        elif status == "skipped":
            print("SKIPPED {0}: did not exist.".format(bucket_name))
        else:
            print("FAILED  {0}: {1}".format(bucket_name, error))

# Final summary
print("\n--- Summary ---")
print("Deleted : {0}".format(len(results["deleted"])))
print("Skipped : {0}".format(len(results["skipped"])))
print("Failed  : {0}".format(len(results["failed"])))
if results["failed"]:
    print("Failed buckets:")
    for b in results["failed"]:
        print("  - {0}".format(b))
if results["skipped"]:
    print(
        "\nNOTE: {0} buckets did not exist at deletion time.".format(
            len(results["skipped"])
        )
    )

#!/bin/bash

set -e

mc alias set minio \
    "$MINIO_ENDPOINT" \
    "$MINIO_ACCESS_KEY" \
    "$MINIO_SECRET_KEY"

if mc ls "minio/$MINIO_BUCKET" > /dev/null 2>&1; then
    echo "Bucket '$MINIO_BUCKET' already exists. Skipping upload."
    exit 0
fi

mc mb "minio/$MINIO_BUCKET"

mc mirror \
    --overwrite \
    /upload \
    "minio/$MINIO_BUCKET"

echo "Files uploaded successfully."
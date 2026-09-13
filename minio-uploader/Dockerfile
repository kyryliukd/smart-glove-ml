FROM quay.io/minio/aistor/mc

COPY upload.sh /upload.sh
RUN chmod +x /upload.sh

COPY files /upload

ENTRYPOINT ["/upload.sh"]
#!/usr/bin/env python3
"""Upload gallery images to Cloudflare R2.

Reads the image keys referenced by gallery/photos.json, finds the matching
resized files written locally by console.html into img/gallery/, and uploads any
that are not already in the bucket. Re-running is idempotent (existing objects
are skipped via a HEAD request).

Pure standard library — signs S3/SigV4 requests itself, no pip installs needed.

Usage:
    python3 gallery/upload.py            # upload anything missing from the bucket
    python3 gallery/upload.py --force    # re-upload everything (overwrite)

Credentials are read from gallery/.env (see gallery/.env.example). That file is
gitignored — secrets never enter this public repo.
"""

import datetime
import hashlib
import hmac
import json
import os
import sys
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(REPO_ROOT, "gallery", ".env")
PHOTOS_JSON = os.path.join(REPO_ROOT, "gallery", "photos.json")
IMG_DIR = os.path.join(REPO_ROOT, "img", "gallery")

REGION = "auto"            # R2 uses the literal region "auto"
SERVICE = "s3"
CONTENT_TYPE = "image/jpeg"
# Images are immutable (keys are content ids) — cache them hard at the edge.
CACHE_CONTROL = "public, max-age=31536000, immutable"


# ---------------------------------------------------------------- config -----
def load_env(path):
    if not os.path.exists(path):
        sys.exit(
            "Missing %s\n"
            "Copy gallery/.env.example to gallery/.env and fill in your R2 "
            "credentials." % path
        )
    env = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    return env


# --------------------------------------------------------------- signing -----
def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _hmac(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret, datestamp):
    k_date = _hmac(("AWS4" + secret).encode("utf-8"), datestamp)
    k_region = hmac.new(k_date, REGION.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, SERVICE.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()


def signed_request(method, url, host, access_key, secret_key, body=b"", extra_headers=None):
    """Build a urllib Request signed with AWS Signature V4."""
    now = datetime.datetime.now(datetime.timezone.utc)
    amzdate = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    path = url.split(host, 1)[1] or "/"
    payload_hash = _sha256(body)

    headers = {
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amzdate,
    }
    if extra_headers:
        headers.update({k.lower(): v for k, v in extra_headers.items()})

    signed_headers = ";".join(sorted(headers))
    canonical_headers = "".join(
        "%s:%s\n" % (k, headers[k]) for k in sorted(headers)
    )
    canonical_request = "\n".join(
        [method, path, "", canonical_headers, signed_headers, payload_hash]
    )

    scope = "%s/%s/%s/aws4_request" % (datestamp, REGION, SERVICE)
    string_to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amzdate, scope, _sha256(canonical_request.encode("utf-8"))]
    )
    signature = hmac.new(
        _signing_key(secret_key, datestamp), string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    authorization = (
        "AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s"
        % (access_key, scope, signed_headers, signature)
    )

    req = urllib.request.Request(url, data=body if method in ("PUT", "POST") else None, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Authorization", authorization)
    return req


# ------------------------------------------------------------------ main -----
def collect_keys():
    with open(PHOTOS_JSON, encoding="utf-8") as fh:
        photos = json.load(fh)
    keys = []
    seen = set()
    for p in photos:
        for field in ("src", "thumb"):
            key = (p.get(field) or "").lstrip("/")
            # tolerate legacy "img/gallery/foo.jpg" paths -> bare key
            key = key.split("/")[-1]
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def object_exists(key, host, access_key, secret_key):
    url = "https://%s/%s" % (host, key)
    req = signed_request("HEAD", url, host, access_key, secret_key)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def put_object(key, data, host, access_key, secret_key):
    url = "https://%s/%s" % (host, key)
    req = signed_request(
        "PUT", url, host, access_key, secret_key, body=data,
        extra_headers={"content-type": CONTENT_TYPE, "cache-control": CACHE_CONTROL},
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError("unexpected status %s for %s" % (resp.status, key))


def main():
    force = "--force" in sys.argv[1:]
    env = load_env(ENV_PATH)
    for required in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        if not env.get(required):
            sys.exit("Missing %s in gallery/.env" % required)

    account_id = env["R2_ACCOUNT_ID"]
    access_key = env["R2_ACCESS_KEY_ID"]
    secret_key = env["R2_SECRET_ACCESS_KEY"]
    bucket = env["R2_BUCKET"]
    public_base = env.get("R2_PUBLIC_BASE", "").rstrip("/")
    # virtual-hosted-style endpoint: <bucket>.<account>.r2.cloudflarestorage.com
    host = "%s.%s.r2.cloudflarestorage.com" % (bucket, account_id)

    keys = collect_keys()
    uploaded = skipped = missing = 0
    for key in keys:
        local = os.path.join(IMG_DIR, key)
        if not os.path.exists(local):
            print("  ! missing local file, skipping: img/gallery/%s" % key)
            missing += 1
            continue
        if not force and object_exists(key, host, access_key, secret_key):
            skipped += 1
            continue
        with open(local, "rb") as fh:
            put_object(key, fh.read(), host, access_key, secret_key)
        where = "%s/%s" % (public_base, key) if public_base else key
        print("  + uploaded %s" % where)
        uploaded += 1

    print(
        "\nDone. %d uploaded, %d already on R2, %d missing locally (of %d keys)."
        % (uploaded, skipped, missing, len(keys))
    )
    if uploaded:
        print("Next: git add gallery/photos.json && git commit")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        sys.exit("R2 request failed: %s %s\n%s" % (e.code, e.reason, e.read().decode("utf-8", "replace")[:500]))

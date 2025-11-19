# DigitalOcean Spaces Integration

This document explains how to configure and use DigitalOcean Spaces storage for automatic file uploads.

## Overview

The model worker now supports automatic uploading of generated 3D models to DigitalOcean Spaces (S3-compatible object storage). This feature is optional and can be enabled via configuration parameters or environment variables.

## Prerequisites

1. A DigitalOcean account
2. A Spaces bucket created in your desired region
3. Spaces API credentials (Access Key and Secret Key)

## Configuration

### Option 1: Environment Variables

Set the following environment variables:

```bash
export DO_SPACES_ENDPOINT="https://nyc3.digitaloceanspaces.com"
export DO_SPACES_REGION="nyc3"
export DO_SPACES_ACCESS_KEY="your_access_key_here"
export DO_SPACES_SECRET_KEY="your_secret_key_here"
export DO_SPACES_BUCKET="your_bucket_name_here"
```

### Option 2: Constructor Parameters

Pass the configuration directly when initializing the ModelWorker:

```python
from model_worker import ModelWorker

worker = ModelWorker(
    model_path='tencent/Hunyuan3D-2.1',
    enable_spaces_upload=True,
    spaces_endpoint='https://nyc3.digitaloceanspaces.com',
    spaces_region='nyc3',
    spaces_access_key='your_access_key',
    spaces_secret_key='your_secret_key',
    spaces_bucket='your_bucket_name'
)
```

## Usage with API Server

When starting the API server, you'll need to pass the Spaces configuration. Update the `api_server.py` to accept and forward these parameters.

## How It Works

1. When `enable_spaces_upload=True`, the worker initializes a boto3 S3 client configured for DigitalOcean Spaces
2. After generating each 3D model file (initial mesh or textured mesh), the file is automatically uploaded to your Spaces bucket
3. Files are uploaded with public-read permissions to the path: `models/{uid}_{type}.glb`
4. The public URL is logged and can be returned in the API response

## Spaces Regions

Available DigitalOcean Spaces regions:
- `nyc3` - New York City, USA
- `sfo2` - San Francisco, USA  
- `ams3` - Amsterdam, Netherlands
- `sgp1` - Singapore
- `fra1` - Frankfurt, Germany

The endpoint URL format is: `https://{region}.digitaloceanspaces.com`

## Getting Your Credentials

1. Log in to your DigitalOcean account
2. Navigate to **API** → **Spaces Keys**
3. Click **Generate New Key**
4. Save your Access Key and Secret Key (the secret key is only shown once)

## Bucket Setup

1. Go to **Spaces** in your DigitalOcean dashboard
2. Click **Create a Space**
3. Choose your region
4. Set your bucket name
5. Configure CDN if desired (optional)
6. Ensure bucket permissions allow public file access

## Security Notes

- Never commit your credentials to version control
- Use environment variables or secure configuration management
- Consider using CDN with your Spaces bucket for better performance
- Set appropriate CORS policies if accessing files from web applications

## Troubleshooting

If uploads fail:
1. Verify your credentials are correct
2. Check that the bucket exists and is accessible
3. Ensure your bucket has public read permissions enabled
4. Check the logs for specific error messages
5. Verify network connectivity to DigitalOcean Spaces

## Dependencies

The integration requires `boto3` which is already included in `requirements.txt`:

```
boto3==1.35.0
```

Install with:
```bash
pip install boto3
```


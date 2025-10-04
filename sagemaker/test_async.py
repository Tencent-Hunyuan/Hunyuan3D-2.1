"""
Test async inference endpoint (no timeout limits)
"""

import json
import base64
import boto3
import time
import sys

def test_async(endpoint_name, image_path):
    """Test async endpoint"""
    
    print(f"Testing async endpoint: {endpoint_name}")
    print(f"Image: {image_path}")
    
    # Get account ID for S3 bucket
    account_id = boto3.client('sts').get_caller_identity()['Account']
    s3_bucket = f'sagemaker-async-{account_id}'
    
    # 1. Read image
    print("\n1. Reading image...")
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    print(f"   Image size: {len(image_bytes)} bytes")
    
    # 2. Prepare payload
    print("\n2. Preparing payload...")
    payload = {
        'image': image_b64,
        'remove_background': True,
        'texture': False,
        'seed': 1234,
        'face_count': 40000,  # Can use high count with async!
        'format': 'stl'
    }
    
    # 3. Upload payload to S3
    print("\n3. Uploading input to S3...")
    s3 = boto3.client('s3')
    
    # Create input key
    timestamp = int(time.time())
    input_key = f'hunyuan3d/inputs/request_{timestamp}.json'
    input_location = f's3://{s3_bucket}/{input_key}'
    
    # Upload payload
    s3.put_object(
        Bucket=s3_bucket,
        Key=input_key,
        Body=json.dumps(payload),
        ContentType='application/json'
    )
    print(f"   Uploaded to: {input_location}")
    
    # 4. Invoke async endpoint
    print("\n4. Invoking async endpoint...")
    print("   (Returns immediately)")
    
    runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
    
    response = runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=input_location,
        ContentType='application/json'
    )
    
    output_location = response['OutputLocation']
    print(f"\n   ✓ Request accepted!")
    print(f"   Output will be at: {output_location}")
    
    # 5. Poll for completion
    print("\n5. Waiting for completion...")
    print("   (Polling S3 every 10 seconds)")
    
    bucket = output_location.split('/')[2]
    key = '/'.join(output_location.split('/')[3:])
    
    for i in range(60):  # Poll for up to 10 minutes
        try:
            # Check if output exists
            s3.head_object(Bucket=bucket, Key=key)
            print(f"\n   ✓ Processing complete! (took ~{i*10} seconds)")
            break
        except:
            print(f"   Attempt {i+1}/60: Still processing...")
            time.sleep(10)
    else:
        print("\n   ⚠️  Timeout waiting for result (10 minutes)")
        print(f"   Check S3 manually: {output_location}")
        return False
    
    # 6. Download result
    print("\n6. Downloading result...")
    response_obj = s3.get_object(Bucket=bucket, Key=key)
    result_data = json.loads(response_obj['Body'].read().decode())
    
    if result_data.get('success'):
        print("\n✅ SUCCESS!")
        print(f"   Faces: {result_data.get('faces', 'N/A')}")
        print(f"   Vertices: {result_data.get('vertices', 'N/A')}")
        
        # Check if files were uploaded to S3 (new method)
        if 'glb_s3_path' in result_data and 'stl_s3_path' in result_data:
            print(f"\n7. Downloading model files from S3...")
            
            # Parse S3 paths
            glb_s3_path = result_data['glb_s3_path']
            stl_s3_path = result_data['stl_s3_path']
            
            # Download GLB
            glb_parts = glb_s3_path.replace('s3://', '').split('/', 1)
            glb_bucket = glb_parts[0]
            glb_key = glb_parts[1]
            
            print(f"   Downloading GLB: {glb_s3_path}")
            try:
                glb_obj = s3.get_object(Bucket=glb_bucket, Key=glb_key)
                glb_bytes = glb_obj['Body'].read()
                
                output_glb = 'output_async.glb'
                with open(output_glb, 'wb') as f:
                    f.write(glb_bytes)
                print(f"   ✓ Saved GLB: {output_glb} ({len(glb_bytes):,} bytes)")
            except Exception as e:
                print(f"   ⚠️  Could not download GLB: {e}")
            
            # Download STL
            stl_parts = stl_s3_path.replace('s3://', '').split('/', 1)
            stl_bucket = stl_parts[0]
            stl_key = stl_parts[1]
            
            print(f"   Downloading STL: {stl_s3_path}")
            try:
                stl_obj = s3.get_object(Bucket=stl_bucket, Key=stl_key)
                stl_bytes = stl_obj['Body'].read()
                
                output_stl = 'output_async.stl'
                with open(output_stl, 'wb') as f:
                    f.write(stl_bytes)
                print(f"   ✓ Saved STL: {output_stl} ({len(stl_bytes):,} bytes)")
            except Exception as e:
                print(f"   ⚠️  Could not download STL: {e}")
            
            print(f"\n   Files saved:")
            print(f"   • output_async.glb - Original high-quality mesh")
            print(f"   • output_async.stl - Processed mesh for 3D printing")
        
        # Old method fallback
        elif 'glb_filename' in result_data and 'stl_filename' in result_data:
            print("\n   ⚠️  Files not uploaded to S3 (using old method)")
            print(f"   Expected: {result_data['glb_filename']}, {result_data['stl_filename']}")
        
        # Base64 fallback
        elif 'model_data' in result_data:
            output_file = 'output_async.stl'
            model_bytes = base64.b64decode(result_data['model_data'])
            with open(output_file, 'wb') as f:
                f.write(model_bytes)
            print(f"\n   Saved to: {output_file}")
            print(f"   File size: {len(model_bytes):,} bytes")
        
        return True
    else:
        print(f"\n❌ FAILED: {result_data.get('error', 'Unknown error')}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python test_async.py ENDPOINT_NAME IMAGE_PATH")
        sys.exit(1)
    
    endpoint = sys.argv[1]
    image = sys.argv[2]
    
    print("="*60)
    print("Async Endpoint Test (No Timeout Limits)")
    print("="*60)
    
    success = test_async(endpoint, image)
    
    print("\n" + "="*60)
    if success:
        print("TEST PASSED ✅")
    else:
        print("TEST FAILED ❌")
    print("="*60)
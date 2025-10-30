"""
Example usage script for the Underwater Image Analysis API.

This script demonstrates how to use the API for image analysis.
"""

import requests
import json
from pathlib import Path


# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_health():
    """Check if the API is healthy and models are loaded."""
    print("=" * 60)
    print("Checking API Health...")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Models Loaded:")
        for model, loaded in health['models_loaded'].items():
            status = "✓" if loaded else "✗"
            print(f"  {status} {model.capitalize()}")
        print(f"Timestamp: {health['timestamp']}")
        return True
    else:
        print(f"Error: API returned status code {response.status_code}")
        return False


def get_config():
    """Get API configuration."""
    print("\n" + "=" * 60)
    print("API Configuration:")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/config")
    
    if response.status_code == 200:
        config = response.json()
        print(f"Max File Size: {config['max_file_size_mb']} MB")
        print(f"Confidence Threshold: {config['confidence_threshold']}")
        print(f"NMS Threshold: {config['nms_threshold']}")
        print(f"Allowed Formats: {', '.join(config['allowed_formats'])}")
    else:
        print(f"Error: Could not retrieve configuration")


def get_detectable_classes():
    """Get list of detectable classes."""
    print("\n" + "=" * 60)
    print("Detectable Classes:")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/classes")
    
    if response.status_code == 200:
        data = response.json()
        classes = data['classes']
        print(f"Total Classes: {data['total_classes']}")
        print("\nClass List:")
        for class_id, class_name in classes.items():
            print(f"  {class_id}: {class_name}")
    else:
        print(f"Error: Could not retrieve classes")


def analyze_image(image_path, confidence_threshold=0.5, nms_threshold=0.45):
    """
    Analyze an underwater image.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Detection confidence threshold
        nms_threshold: NMS IoU threshold
    """
    print("\n" + "=" * 60)
    print(f"Analyzing Image: {image_path}")
    print("=" * 60)
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}")
        return
    
    # Prepare request
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold
        }
        
        print(f"Uploading image...")
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n✓ Analysis Successful!")
        print(f"Request ID: {result['request_id']}")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print(f"Message: {result['message']}")
        
        # Image dimensions
        if result.get('image_dimensions'):
            dims = result['image_dimensions']
            print(f"\nImage Dimensions:")
            print(f"  Original: {dims['original']['width']}x{dims['original']['height']}")
            print(f"  Enhanced: {dims['enhanced']['width']}x{dims['enhanced']['height']}")
        
        # Detections
        detections = result['detections']
        print(f"\nDetections: {len(detections)}")
        
        if detections:
            print("\nDetailed Results:")
            for i, det in enumerate(detections, 1):
                print(f"\n  Detection {i}:")
                print(f"    Class: {det['class_name']}")
                print(f"    Confidence: {det['confidence']:.3f}")
                print(f"    Bounding Box: {det['bbox']}")
        else:
            print("  No objects detected in the image.")
        
        # Annotated image URL
        annotated_url = result['annotated_image_url']
        full_url = f"{API_BASE_URL}{annotated_url}"
        print(f"\nAnnotated Image URL: {full_url}")
        print(f"View in browser: {full_url}")
        
        # Optionally download the annotated image
        download = input("\nDownload annotated image? (y/n): ").lower()
        if download == 'y':
            download_annotated_image(annotated_url, result['request_id'])
    
    elif response.status_code == 400:
        error = response.json()
        print(f"\n✗ Validation Error:")
        print(f"  {error.get('detail', 'Unknown error')}")
    
    elif response.status_code == 503:
        print(f"\n✗ Service Unavailable:")
        print(f"  Models not loaded. Please check the server.")
    
    else:
        print(f"\n✗ Error: API returned status code {response.status_code}")
        try:
            error = response.json()
            print(f"  {error.get('detail', 'Unknown error')}")
        except:
            print(f"  {response.text}")


def download_annotated_image(image_url, request_id):
    """Download the annotated image."""
    response = requests.get(f"{API_BASE_URL}{image_url}")
    
    if response.status_code == 200:
        output_path = f"annotated_{request_id}.jpg"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✓ Annotated image saved to: {output_path}")
    else:
        print(f"✗ Failed to download annotated image")


def analyze_batch(image_paths, confidence_threshold=0.5):
    """
    Analyze multiple images in batch.
    
    Args:
        image_paths: List of image file paths
        confidence_threshold: Detection confidence threshold
    """
    print("\n" + "=" * 60)
    print(f"Batch Analysis: {len(image_paths)} images")
    print("=" * 60)
    
    # Prepare files
    files = []
    for path in image_paths:
        if Path(path).exists():
            files.append(('files', open(path, 'rb')))
        else:
            print(f"Warning: File not found: {path}")
    
    if not files:
        print("Error: No valid files to analyze")
        return
    
    # Send request
    data = {'confidence_threshold': confidence_threshold}
    response = requests.post(
        f"{API_BASE_URL}/analyze-batch",
        files=files,
        data=data
    )
    
    # Close files
    for _, f in files:
        f.close()
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Batch Analysis Complete!")
        print(f"Total Images: {result['total']}")
        print(f"Processed: {result['processed']}")
        
        # Show results for each image
        for i, img_result in enumerate(result['results'], 1):
            if isinstance(img_result, dict) and img_result.get('success'):
                print(f"\nImage {i}:")
                print(f"  Detections: {len(img_result['detections'])}")
                print(f"  Processing Time: {img_result['processing_time']:.2f}s")
            else:
                print(f"\nImage {i}: Failed")
    else:
        print(f"✗ Batch analysis failed with status code {response.status_code}")


def interactive_mode():
    """Interactive mode for testing the API."""
    print("\n" + "=" * 60)
    print("UNDERWATER IMAGE ANALYSIS API - INTERACTIVE MODE")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("  1. Check API Health")
        print("  2. Get Configuration")
        print("  3. Get Detectable Classes")
        print("  4. Analyze Single Image")
        print("  5. Analyze Batch of Images")
        print("  6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            check_health()
        
        elif choice == '2':
            get_config()
        
        elif choice == '3':
            get_detectable_classes()
        
        elif choice == '4':
            image_path = input("Enter image path: ").strip()
            confidence = input("Confidence threshold (default 0.5): ").strip()
            confidence = float(confidence) if confidence else 0.5
            analyze_image(image_path, confidence_threshold=confidence)
        
        elif choice == '5':
            paths = input("Enter image paths (comma-separated): ").strip()
            image_paths = [p.strip() for p in paths.split(',')]
            confidence = input("Confidence threshold (default 0.5): ").strip()
            confidence = float(confidence) if confidence else 0.5
            analyze_batch(image_paths, confidence_threshold=confidence)
        
        elif choice == '6':
            print("\nExiting...")
            break
        
        else:
            print("\nInvalid choice. Please try again.")


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("UNDERWATER IMAGE ANALYSIS API - EXAMPLE USAGE")
    print("=" * 60)
    print(f"\nAPI URL: {API_BASE_URL}")
    print("\nMake sure the API server is running!")
    print("Start server with: python app/main.py")
    
    # Check if API is available
    try:
        check_health()
        get_config()
        get_detectable_classes()
        
        # Ask for interactive mode
        print("\n")
        choice = input("Enter interactive mode? (y/n): ").lower()
        if choice == 'y':
            interactive_mode()
        else:
            # Example single analysis
            print("\nFor single image analysis, use:")
            print("  python example_usage.py <image_path>")
            
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("  Make sure the API server is running at", API_BASE_URL)
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    import sys
    
    # If image path provided as argument, analyze it
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        
        check_health()
        analyze_image(image_path, confidence_threshold=confidence)
    else:
        # Run interactive mode
        main()

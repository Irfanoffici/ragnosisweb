#!/usr/bin/env python3
"""
Quick test script for ClaraGPT
"""
import requests
import json
import time

def test_clara_gpt():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing ClaraGPT Medical Assistant...")
    
    # Test 1: Health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Ask a medical question
    test_questions = [
        "What is hypertension?",
        "Tell me about diabetes symptoms",
        "How to prevent influenza?"
    ]
    
    for question in test_questions:
        try:
            print(f"\n🔍 Testing: '{question}'")
            start_time = time.time()
            
            response = requests.post(
                f"{base_url}/ask",
                json={"question": question},
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success in {response_time:.2f}s")
                print(f"   Answer preview: {data['answer'][:100]}...")
                print(f"   Confidence: {data['confidence']}")
                print(f"   Sources: {len(data['sources'])}")
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_clara_gpt()

import requests
import sys
import json
import time
from datetime import datetime

class MeetingAssistantAPITester:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)}")
                    return True, response_data
                except:
                    print(f"   Response: {response.text}")
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error Response: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"   Error Response: {response.text}")
                return False, {}

        except requests.exceptions.ConnectionError as e:
            print(f"❌ Failed - Connection Error: {str(e)}")
            print("   Backend server may not be running or accessible")
            return False, {}
        except requests.exceptions.Timeout as e:
            print(f"❌ Failed - Timeout Error: {str(e)}")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test health check endpoint"""
        return self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )

    def test_start_meeting(self):
        """Test start meeting endpoint"""
        return self.run_test(
            "Start Meeting",
            "POST",
            "api/meeting/start",
            200
        )

    def test_stop_meeting(self):
        """Test stop meeting endpoint"""
        return self.run_test(
            "Stop Meeting",
            "POST",
            "api/meeting/stop",
            200
        )

    def test_set_agenda(self):
        """Test set agenda endpoint"""
        agenda_data = {
            "items": [
                "Welcome and introductions",
                "Project status update",
                "Budget discussion",
                "Next steps and action items"
            ]
        }
        return self.run_test(
            "Set Meeting Agenda",
            "POST",
            "api/meeting/agenda",
            200,
            data=agenda_data
        )

    def test_invalid_agenda(self):
        """Test agenda endpoint with invalid data"""
        return self.run_test(
            "Set Invalid Agenda",
            "POST",
            "api/meeting/agenda",
            200,  # Backend should handle gracefully
            data={}
        )

def main():
    print("🚀 Starting Meeting Assistant API Tests")
    print("=" * 50)
    
    # Setup
    tester = MeetingAssistantAPITester()
    
    # Test sequence
    print("\n📋 Running API Endpoint Tests...")
    
    # 1. Health check - most basic test
    health_success, health_data = tester.test_health_check()
    if not health_success:
        print("\n❌ Health check failed - Backend may not be running")
        print("   Please check if the backend server is accessible")
        return 1
    
    # 2. Test agenda setting
    agenda_success, _ = tester.test_set_agenda()
    
    # 3. Test meeting start
    start_success, _ = tester.test_start_meeting()
    
    # 4. Wait a moment to simulate meeting activity
    if start_success:
        print("\n⏳ Waiting 2 seconds to simulate meeting activity...")
        time.sleep(2)
    
    # 5. Test meeting stop
    stop_success, _ = tester.test_stop_meeting()
    
    # 6. Test invalid agenda
    invalid_agenda_success, _ = tester.test_invalid_agenda()
    
    # Print final results
    print("\n" + "=" * 50)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All API tests passed!")
        return 0
    else:
        failed_tests = tester.tests_run - tester.tests_passed
        print(f"⚠️  {failed_tests} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
import asyncio
import websockets
import json
import sys

async def test_websocket_connection():
    """Test WebSocket connection and message handling"""
    uri = "ws://localhost:8001/api/ws"
    
    try:
        print("🔌 Testing WebSocket Connection...")
        print(f"   Connecting to: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Send a ping message
            ping_message = {"type": "ping", "timestamp": "test"}
            await websocket.send(json.dumps(ping_message))
            print("📤 Sent ping message")
            
            # Wait for any responses (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"📥 Received response: {response}")
            except asyncio.TimeoutError:
                print("⏰ No response received (timeout - this is expected)")
            
            # Test connection stability
            await asyncio.sleep(2)
            
            # Send another message
            test_message = {"type": "test", "data": "connection_test"}
            await websocket.send(json.dumps(test_message))
            print("📤 Sent test message")
            
            # Wait a bit more
            await asyncio.sleep(1)
            
            print("✅ WebSocket connection test completed successfully")
            return True
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ WebSocket connection refused - Backend may not be running")
        return False
    except Exception as e:
        print(f"❌ WebSocket error: {str(e)}")
        return False

async def main():
    print("🚀 Starting WebSocket Integration Test")
    print("=" * 50)
    
    success = await test_websocket_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 WebSocket integration test passed!")
        return 0
    else:
        print("❌ WebSocket integration test failed!")
        return 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)
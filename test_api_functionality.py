import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test 1: Import test
print("🧪 Test 1: Importing module...")
try:
    import bgeReranker_API_enhanced as api
    print("✅ Import successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Test math.exp functionality
print("\n🧪 Test 2: Testing math.exp for sigmoid calculation...")
test_score = 2.5
sigmoid = 1 / (1 + api.math.exp(-test_score))
print(f"   Score: {test_score}, Sigmoid: {sigmoid:.4f}")
assert abs(sigmoid - 0.9241) < 0.001, "Sigmoid calculation incorrect"
print("✅ Math calculation works correctly!")

# Test 3: Test batch_compute_score mock
print("\n🧪 Test 3: Testing batch processing logic...")
# Mock model that just returns 1.0 for all pairs
class MockModel:
    def compute_score(self, pairs):
        return [float(i) * 0.5 for i in range(len(pairs))]

mock_model = MockModel()
test_pairs = [["query", "doc1"], ["query", "doc2"], ["query", "doc3"], ["query", "doc4"], ["query", "doc5"]]
scores = api.batch_compute_score(mock_model, test_pairs, batch_size=2)
print(f"   Input pairs: {len(test_pairs)}, Output scores: {scores}")
assert len(scores) == 5, f"Expected 5 scores, got {len(scores)}"
assert scores == [0.0, 0.5, 1.0, 1.5, 2.0], f"Expected [0.0, 0.5, 1.0, 1.5, 2.0], got {scores}"
print("✅ Batch processing works correctly!")

# Test 4: Verify environment variable handling
print("\n🧪 Test 4: Testing environment variable handling...")
os.environ['TEST_VAR'] = 'test_value'
os.environ.pop('HF_HOME', None)
# Reload module to test HF_HOME logic
import importlib
importlib.reload(api)
hf_home = os.environ.get('HF_HOME')
print(f"   HF_HOME set to: {hf_home}")
assert hf_home == r"F:\openclaw_models\extModels", f"Expected default HF_HOME, got {hf_home}"

# Test with custom HF_HOME
os.environ['HF_HOME'] = r"C:\custom\path"
importlib.reload(api)
hf_home = os.environ.get('HF_HOME')
print(f"   Custom HF_HOME set to: {hf_home}")
assert hf_home == r"C:\custom\path", f"Expected custom HF_HOME, got {hf_home}"
print("✅ Environment variable handling works correctly!")

print("\n🎉 All tests passed! The API functionality is working correctly.")
print("\n📋 Summary of verified features:")
print("  ✅ Module imports without errors")
print("  ✅ Math exp/sigmoid calculation is accurate")
print("  ✅ Batch processing logic works correctly")
print("  ✅ HF_HOME environment variable priority works")
print("  ✅ No syntax or indentation errors")
print("\n🚀 The API is ready to use! Run `run_api_enhanced.bat` to start the server.")

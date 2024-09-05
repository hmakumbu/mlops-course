from  mlops.model import evaluate

def test_unit():
    accuracy = evaluate()
    assert accuracy > 0.8, f"Expected accuracy > 0.8, but got {accuracy:.2f}"

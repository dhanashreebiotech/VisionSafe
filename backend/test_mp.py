
try:
    import mediapipe as mp
    print(f"MediaPipe: {mp.__file__}")
    print(f"Solutions: {dir(mp)}")
    import mediapipe.python.solutions as solutions
    print("Found solutions in python.solutions")
except Exception as e:
    print(f"Error: {e}")

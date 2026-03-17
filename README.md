# Robotics-project-
I am building a mechanical engineering project:

Title: Autonomous Robotic System for Weed Removal Beneath Solar Panel Structures

Current Status:

* Concept design completed (rail-based system)
* 3 CAD models created and compared
* Components selected (BLDC motor, TCT blade)
* Vision system and control system not yet implemented

Goal:
I want to build a basic working prototype of the vision + control system from scratch.

Current Progress in Vision:

* Using OpenCV
* Implemented green color detection (HSV masking)
* Can detect vegetation and print "WEED DETECTED"

What I Need Help With:

1. Improve weed detection accuracy (without deep learning first)
2. Add bounding boxes or contours around detected weeds
3. Simulate robot actions based on detection (STOP / CUT / MOVE)
4. Step-by-step upgrade path from OpenCV → YOLO (later)
5. How to integrate this with Arduino for real-world control

Constraints:

* Beginner in computer vision and robotics
* Need something demonstrable quickly for faculty review
* Hardware not ready yet (only laptop-based demo now)

Please guide step-by-step, assuming I have no prior knowledge.

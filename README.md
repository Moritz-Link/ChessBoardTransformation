Chessboard Edge Detection with YOLO

This project implements a key point detection model using YOLO to identify the edges of a chessboard. The primary goal is to detect the four corners of the board in an image, even if the chessboard is rotated, and then transform the view into a top-down (bird’s-eye) perspective.

✨ Features

Custom dataset: I created and labeled a dataset specifically for this task, focusing on chessboard corner detection.

YOLO-based detection: The model leverages YOLO for efficient and accurate key point detection.

Perspective transformation: Once the corners are detected, the board is geometrically transformed into a rectified bird’s-eye view.

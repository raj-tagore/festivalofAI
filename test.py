import cv2

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Get the size of the frame
        height, width, _ = frame.shape
        print(f"Frame size: {width}x{height}")

        # Display the frame
        cv2.imshow("Webcam Feed", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()